from __future__ import annotations

"""
将俯视/伪正射图像赋予伪地理坐标：
- 输入：普通影像文件、左下角经纬度(lat_min, lon_min)、像素分辨率（米/像素或度/像素）、以及可选的目标CRS。
- 输出：
  1) 若安装了 rasterio：直接写GeoTIFF（推荐）。
  2) 否则：写出World File(+.wld)与WGS84的.prj文件，原图不改名。

注意：经纬度是角度单位，不建议直接用“米/像素”一步写度制仿射。若你仅有米/像素且范围不大，可近似按纬度缩放换算：
  dlon_deg_per_m ≈ 1 / (111320 * cos(lat_mid))
  dlat_deg_per_m ≈ 1 / 110574
请理解这是近似。若需严格米制，请改用投影坐标（UTM/本地ENU）或先重投影。
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np


def _deg_per_meter(lat_deg: float) -> tuple[float, float]:
	lat_rad = math.radians(lat_deg)
	dlon = 1.0 / (111320.0 * math.cos(max(1e-8, lat_rad)))
	dlat = 1.0 / 110574.0
	return dlat, dlon


def write_worldfile(base_path: Path, px_size_x: float, px_size_y: float, ulx: float, uly: float) -> None:
	"""写入ESRI World File（六参数，无旋转）。扩展名：.wld。

	注意：C、F应为左上角像元中心坐标，因此若ulx/uly是左上角影像边界坐标，需要加半个像元偏移。
	这里我们传入的ulx/uly已经是左上角像元中心坐标。
	"""
	wld = base_path.with_suffix(".wld")
	# A, D, B, E, C, F  (A=px size x, E=-px size y), C/F=UL像元中心
	content = "\n".join([
		f"{px_size_x:.12f}",
		"0.0",
		"0.0",
		f"{-px_size_y:.12f}",
		f"{ulx:.12f}",
		f"{uly:.12f}",
	])
	wld.write_text(content)


def write_wgs84_prj(base_path: Path) -> None:
	prj = base_path.with_suffix(".prj")
	prj.write_text(
		'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
		'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
	)



def save_geotiff(input_img: Path, output_tif: Path, ulx: float, uly: float, px_size_x: float, px_size_y: float, crs_epsg: int = 4326) -> None:
	try:
		import rasterio
		rasterio_env_ok = True
	except Exception:
		rasterio_env_ok = False
	if not rasterio_env_ok:
		raise RuntimeError("未安装rasterio，无法直接写GeoTIFF。")
	with Image.open(input_img) as im:
		w, h = im.size
		arr = im.convert("RGB")
		arr = np.array(arr)

	from rasterio.transform import Affine
	from rasterio.crs import CRS

	# Affine: X = A * col + B * row + C, Y = D * col + E * row + F
	# rasterio以像元左上角为原点，因此C/F应为左上角像元左上角坐标。
	# 我们传入的ulx/uly为左上角像元中心坐标，需要减去/加上半个像元得到边界。
	C = ulx - px_size_x * 0.5
	F = uly + px_size_y * 0.5  # E为负，向下为负北向
	transform = Affine(px_size_x, 0.0, C, 0.0, -px_size_y, F)
	profile = {
		"driver": "GTiff",
		"height": arr.shape[0],
		"width": arr.shape[1],
		"count": 3,
		"dtype": arr.dtype,
		"crs": CRS.from_epsg(crs_epsg),
		"transform": transform,
	}
	with rasterio.open(output_tif, "w", **profile) as dst:
		for i in range(3):
			dst.write(arr[:, :, i], i + 1)


def main():
	parser = argparse.ArgumentParser(description="为俯视/伪正射影像赋予伪地理坐标（WGS84，经纬度）")
	parser.add_argument("image", type=str, help="输入影像路径（如jpg/png）")
	parser.add_argument("lat_min", type=float, help="左下角维度（南北，度）")
	parser.add_argument("lon_min", type=float, help="左下角经度（东西，度）")
	parser.add_argument("res", type=float, help="像素分辨率，单位：米/像素（近似转换为度）")
	parser.add_argument("out", type=str, help="输出GeoTIFF路径（若无rasterio则输出worldfile与prj）")
	parser.add_argument("--lat-mid", type=float, default=None, help="用于米->度换算的参考纬度（默认取lat_min近似）")

	args = parser.parse_args()
	img = Path(args.image)
	out = Path(args.out)
	with Image.open(img) as im:
		w, h = im.size

	lat_ref = args.lat_mid if args.lat_mid is not None else args.lat_min
	dlat_deg_per_m, dlon_deg_per_m = _deg_per_meter(lat_ref)
	# 像素尺寸（度/像素）
	px_dlat = dlat_deg_per_m * args.res
	px_dlon = dlon_deg_per_m * args.res

	# 左上角经纬度（像元中心坐标）；GeoTIFF内部会转为左上角边界坐标
	ulx = args.lon_min
	uly = args.lat_min + h * px_dlat

	try:
		save_geotiff(img, out, ulx=ulx, uly=uly, px_size_x=px_dlon, px_size_y=px_dlat, crs_epsg=4326)
		print(f"写入GeoTIFF: {out}")
	except Exception as e:
		print(f"rasterio不可用或写入失败，改写WorldFile与PRJ。原因: {e}")
		# 写WorldFile到源影像同名（.wld），不改图像数据
		write_worldfile(img, px_size_x=px_dlon, px_size_y=px_dlat, ulx=ulx, uly=uly)
		write_wgs84_prj(img)
		print(f"已写入: {img.with_suffix('.wld')} 与 {img.with_suffix('.prj')} (WGS84)。")

	# 另外写出一个json记录变换参数，便于后续复用
	meta = {
		"crs": "EPSG:4326",
		"ulx": ulx,
		"uly": uly,
		"px_size_x_deg": px_dlon,
		"px_size_y_deg": px_dlat,
		"width": w,
		"height": h,
	}
	Path(str(out) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
	print(f"已保存元数据: {str(out)}.json")


if __name__ == "__main__":
	main()


