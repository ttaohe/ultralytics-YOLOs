from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .geope import (
    encode_view_geocoords,
    visualize_lonlat,
    _load_H_json,
    _open_raster_transform,
    _get_raster_size,
)


def main():
    parser = argparse.ArgumentParser(description="基于H与GeoTIFF为视角图像生成经纬度位置编码，并输出可视化")
    parser.add_argument("view", type=str, help="视角图像路径")
    parser.add_argument("H_view2ortho", type=str, help="view->ortho 的H json")
    parser.add_argument("base_geotiff", type=str, help="带地理参照的正射GeoTIFF")
    parser.add_argument("out_dir", type=str, help="输出目录")
    parser.add_argument("--stride", type=int, default=4, help="下采样步长，用于加速")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lon_grid, lat_grid, mask = encode_view_geocoords(
        args.view, args.H_view2ortho, args.base_geotiff, stride=args.stride
    )
    # 保存npy
    (np.save(out / "lon.npy", lon_grid), np.save(out / "lat.npy", lat_grid), np.save(out / "mask.npy", mask))

    # 可视化
    vis = visualize_lonlat(lon_grid, lat_grid, mask)
    try:
        import cv2
        cv2.imwrite(str(out / "geope.png"), vis)
    except Exception:
        from PIL import Image
        Image.fromarray(vis[:, :, ::-1]).save(out / "geope.png")

    # 导出lon/lat灰度图，便于在GIS中核对
    try:
        import cv2
        lon_norm = (lon_grid - np.nanmin(lon_grid[mask])) / (max(np.nanmax(lon_grid[mask]) - np.nanmin(lon_grid[mask]), 1e-12))
        lat_norm = (lat_grid - np.nanmin(lat_grid[mask])) / (max(np.nanmax(lat_grid[mask]) - np.nanmin(lat_grid[mask]), 1e-12))
        cv2.imwrite(str(out / "lon.png"), (lon_norm * 255).astype(np.uint8))
        cv2.imwrite(str(out / "lat.png"), (lat_norm * 255).astype(np.uint8))
    except Exception:
        pass

    # 额外：将视角图像透视到正射尺寸，便于校验H
    try:
        import cv2
        H = _load_H_json(args.H_view2ortho)
        # 读取正射尺寸
        try:
            import rasterio
            with rasterio.open(args.base_geotiff) as ds:
                dst_size = (ds.width, ds.height)
        except Exception:
            from PIL import Image as _PILImage
            with _PILImage.open(args.base_geotiff) as im0:
                dst_size = im0.size
        # 读取视角图并warp
        img_view = cv2.imread(str(args.view), cv2.IMREAD_COLOR)
        if img_view is not None:
            warped = cv2.warpPerspective(img_view, H, dst_size, flags=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / "warped_view_to_ortho.png"), warped)
            # 生成NoData掩码：仅保留“由源像素映射到正射”的区域，其余为NoData
            mask_src = np.ones((img_view.shape[0], img_view.shape[1]), dtype=np.uint8) * 255
            alpha = cv2.warpPerspective(mask_src, H, dst_size, flags=cv2.INTER_NEAREST)
            # 写GeoTIFF（三波段RGB + GDAL内部mask作为NoData）
            try:
                import rasterio
                with rasterio.open(args.base_geotiff) as ds:
                    profile = ds.profile
                    profile.update({"count": 3, "dtype": "uint8", "photometric": "RGB"})
                    tif_path = out / "warped_view_to_ortho.tif"
                    with rasterio.open(tif_path, "w", **profile) as dst:
                        dst.write(warped[:, :, 0], 1)
                        dst.write(warped[:, :, 1], 2)
                        dst.write(warped[:, :, 2], 3)
                        # 设置GDAL内部mask为有效区域（alpha>0），其余作为NoData，可在QGIS中透明
                        dst.write_mask(alpha)

                # 使用GDAL Python API继续处理：裁剪到有效mask范围、压缩并生成金字塔
                try:
                    from osgeo import gdal
                    ds = gdal.Open(str(tif_path))
                    if ds is not None:
                        band = ds.GetRasterBand(1)
                        mband = band.GetMaskBand()
                        m = mband.ReadAsArray()
                        ys, xs = (m > 0).nonzero()
                        if ys.size > 0 and xs.size > 0:
                            xoff, yoff = int(xs.min()), int(ys.min())
                            xsize = int(xs.max() - xoff + 1)
                            ysize = int(ys.max() - yoff + 1)
                            cropped_path = out / "warped_view_to_ortho_cropped.tif"
                            translate_opts = gdal.TranslateOptions(
                                format="GTiff",
                                creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
                                srcWin=[xoff, yoff, xsize, ysize],
                            )
                            gdal.Translate(str(cropped_path), ds, options=translate_opts)
                            # 构建金字塔
                            cds = gdal.Open(str(cropped_path), gdal.GA_Update)
                            if cds is not None:
                                gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
                                cds.BuildOverviews("NEAREST", [2, 4, 8, 16])
                                cds = None
                            print(f"Saved cropped: {cropped_path}")
                        ds = None
                except Exception:
                    pass
            except Exception:
                pass
            # 反映射：将正射warp结果用 H^{-1} 投回原图尺寸，便于对比几何一致性
            try:
                H_inv = np.linalg.inv(H)
                h0, w0 = img_view.shape[0], img_view.shape[1]
                back = cv2.warpPerspective(warped, H_inv, (w0, h0), flags=cv2.INTER_LINEAR)
                cv2.imwrite(str(out / "warped_ortho_back_to_view.png"), back)
                # 同步反映射有效区域mask，输出带透明通道的PNG（可直观叠加原图）
                alpha_back = cv2.warpPerspective(alpha, H_inv, (w0, h0), flags=cv2.INTER_NEAREST)
                bgra = np.dstack([back, alpha_back])
                cv2.imwrite(str(out / "warped_ortho_back_to_view_rgba.png"), bgra)
            except Exception:
                pass
    except Exception:
        pass

    print(f"Saved: {out}/geope.png and lon/lat/mask .npy")

    # 基准正射影像的地理位置编码可视化（用于对齐对比）
    try:
        import numpy as _np
        T, _ = _open_raster_transform(args.base_geotiff)
        Wb, Hb = _get_raster_size(args.base_geotiff)
        gy_b, gx_b = _np.mgrid[0: Hb: args.stride, 0: Wb: args.stride]
        cols_b = gx_b.astype(_np.float64)
        rows_b = gy_b.astype(_np.float64)
        # 像元中心
        xy1 = _np.stack([cols_b + 0.5, rows_b + 0.5, _np.ones_like(cols_b)], axis=-1)
        xy = xy1 @ T.T
        lon_b = xy[..., 0]
        lat_b = xy[..., 1]
        mask_b = _np.ones_like(lon_b, dtype=bool)
        # 保存
        _np.save(out / "lon_base.npy", lon_b)
        _np.save(out / "lat_base.npy", lat_b)
        _np.save(out / "mask_base.npy", mask_b)
        vis_b = visualize_lonlat(lon_b, lat_b, mask_b)
        try:
            import cv2
            cv2.imwrite(str(out / "geope_base.png"), vis_b)
            # 灰度导出
            lon_bn = (lon_b - _np.nanmin(lon_b)) / max(_np.nanmax(lon_b) - _np.nanmin(lon_b), 1e-12)
            lat_bn = (lat_b - _np.nanmin(lat_b)) / max(_np.nanmax(lat_b) - _np.nanmin(lat_b), 1e-12)
            cv2.imwrite(str(out / "lon_base.png"), (lon_bn * 255).astype(_np.uint8))
            cv2.imwrite(str(out / "lat_base.png"), (lat_bn * 255).astype(_np.uint8))
        except Exception:
            from PIL import Image as _Image
            _Image.fromarray(vis_b[:, :, ::-1]).save(out / "geope_base.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()


