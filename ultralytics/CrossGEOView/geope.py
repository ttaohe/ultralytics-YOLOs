from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _load_H_json(path: str | Path) -> np.ndarray:
	obj = json.loads(Path(path).read_text())
	H = obj["H"] if isinstance(obj, dict) and "H" in obj else obj
	return np.asarray(H, dtype=np.float64).reshape(3, 3)


def _open_raster_transform(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回(affine_matrix_3x3, crs_str)。若无rasterio，则尝试读取同名.json元数据。"""
    try:
        import rasterio
        with rasterio.open(path) as src:
            T = np.array(
                [
                    [src.transform.a, src.transform.b, src.transform.c],
                    [src.transform.d, src.transform.e, src.transform.f],
                    [0.0, 0.0, 1.0],
                ]
            )
            crs = str(src.crs) if src.crs is not None else ""
            return T, crs
    except Exception:
        meta_path = Path(str(path) + ".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"无法读取{path}的仿射参数，且未找到元数据 {meta_path}")
        m = json.loads(meta_path.read_text())
        # 使用像元中心坐标与像素尺寸组装仿射（左上角边界）
        A = float(m.get("px_size_x_deg"))  # 经度方向度/像素
        py = float(m.get("px_size_y_deg"))  # 纬度方向度/像素（正值）
        C = float(m.get("ulx")) - A * 0.5
        F = float(m.get("uly")) + py * 0.5
        T = np.array([[A, 0.0, C], [0.0, -py, F], [0.0, 0.0, 1.0]], dtype=np.float64)
        return T, m.get("crs", "EPSG:4326")


def _get_raster_size(path: str | Path) -> Tuple[int, int]:
    """返回(宽W, 高H)。"""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.width, ds.height
    except Exception:
        with Image.open(path) as im0:
            return im0.size


def _pixels_to_geo(T: np.ndarray, cols: np.ndarray, rows: np.ndarray, as_center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
	"""像素(col,row) -> 地理坐标(x=lon,y=lat)。as_center=True表示用像元中心。"""
	if as_center:
		cols = cols + 0.5
		rows = rows + 0.5
	xy1 = np.stack([cols, rows, np.ones_like(cols)], axis=-1)  # [...,3]
	xy = xy1 @ T.T  # [...,3]
	return xy[..., 0], xy[..., 1]


def encode_view_geocoords(
	view_img_path: str | Path,
	H_view2ortho_json: str | Path,
	base_geotiff_path: str | Path,
	stride: int = 4,
	return_full: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为视角图像生成经纬度位置编码（基于H与GeoTIFF）。

    Returns: (lon_grid, lat_grid, mask)
    - lon_grid/lat_grid: [h', w'] 栅格（h'=ceil(H/stride), w'=ceil(W/stride)）
    - mask: 有效像元（H投影到正射范围内）的布尔掩码
    """
    H = _load_H_json(H_view2ortho_json)
    T, _ = _open_raster_transform(base_geotiff_path)
    Wo, Ho = _get_raster_size(base_geotiff_path)

    with Image.open(view_img_path) as im:
        W, Himg = im.size

    gy, gx = np.mgrid[0:Himg:stride, 0:W:stride]
    gy = gy.astype(np.float64)
    gx = gx.astype(np.float64)
    N = gy.size
    pts = np.stack([gx.reshape(-1), gy.reshape(-1), np.ones(N)], axis=1)  # [N,3]
    # view -> ortho 像素
    po = (pts @ H.T)
    po = po[:, :2] / (po[:, 2:3] + 1e-12)  # [N,2] col,row
    cols = po[:, 0]
    rows = po[:, 1]
    # 在正射栅格内的有效点
    valid = (
        np.isfinite(cols)
        & np.isfinite(rows)
        & (cols >= 0)
        & (rows >= 0)
        & (cols < Wo)
        & (rows < Ho)
    )
    # 地理坐标
    lon, lat = _pixels_to_geo(T, cols, rows, as_center=True)
    valid &= np.isfinite(lon) & np.isfinite(lat)
    # 将无效点置为NaN，避免异常范围影响可视化
    lon[~valid] = np.nan
    lat[~valid] = np.nan
    lon_grid = lon.reshape(gy.shape)
    lat_grid = lat.reshape(gy.shape)
    mask = valid.reshape(gy.shape)
    return lon_grid, lat_grid, mask


def visualize_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""将经纬度编码可视化为RGB图（按局部min-max归一化）。"""
	import cv2
	lon = lon_grid.copy()
	lat = lat_grid.copy()
	# 局部归一化
	lo_min, lo_max = np.nanmin(lon[mask]), np.nanmax(lon[mask])
	la_min, la_max = np.nanmin(lat[mask]), np.nanmax(lat[mask])
	lon_n = (lon - lo_min) / (max(lo_max - lo_min, 1e-12))
	lat_n = (lat - la_min) / (max(la_max - la_min, 1e-12))
	vis = np.zeros((lon.shape[0], lon.shape[1], 3), dtype=np.float32)
	vis[..., 0] = lon_n  # R
	vis[..., 1] = lat_n  # G
	vis[..., 2] = 0.5
	vis[~mask] = 0
	vis = (vis * 255).clip(0, 255).astype(np.uint8)
	vis = cv2.applyColorMap(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
	return vis


