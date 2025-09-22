from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geope import _load_H_json, _open_raster_transform


def main():
	parser = argparse.ArgumentParser(description="按H将视角影像warp到正射网格，并写出带地理参照的GeoTIFF")
	parser.add_argument("view", type=str, help="视角影像")
	parser.add_argument("H_view2ortho", type=str, help="view->ortho 的H json")
	parser.add_argument("base_geotiff", type=str, help="正射GeoTIFF，提供仿射与尺寸")
	parser.add_argument("out_tif", type=str, help="输出GeoTIFF路径")
	args = parser.parse_args()

	H = _load_H_json(args.H_view2ortho)
	T, _ = _open_raster_transform(args.base_geotiff)

	# 读取正射尺寸
	try:
		import rasterio
		with rasterio.open(args.base_geotiff) as ds:
			W, Hh = ds.width, ds.height
	except Exception:
		with Image.open(args.base_geotiff) as im0:
			W, Hh = im0.size

	# warp
	import cv2
	img_view = cv2.imread(str(args.view), cv2.IMREAD_COLOR)
	warped = cv2.warpPerspective(img_view, H, (W, Hh), flags=cv2.INTER_LINEAR)

	# 写GeoTIFF（使用base的仿射和CRS）
	try:
		import rasterio
		from rasterio.crs import CRS
		from rasterio.transform import Affine
		with rasterio.open(args.base_geotiff) as ds:
			profile = ds.profile
			profile.update({"count": 3, "dtype": "uint8"})
			with rasterio.open(args.out_tif, "w", **profile) as dst:
				for i in range(3):
					dst.write(warped[:, :, i], i + 1)
		print(f"Saved GeoTIFF: {args.out_tif}")
	except Exception as e:
		# 回退为写普通PNG和伴随wld/prj（WGS84）
		out_png = str(Path(args.out_tif).with_suffix(".png"))
		cv2.imwrite(out_png, warped)
		print(f"rasterio不可用，已写PNG: {out_png}（可配合原base的wld/prj使用）")


if __name__ == "__main__":
	main()


