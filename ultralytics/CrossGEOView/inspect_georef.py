from __future__ import annotations

import argparse
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description="检查GeoTIFF/栅格的CRS、仿射参数与四角坐标")
	parser.add_argument("raster", type=str, help="输入GeoTIFF路径")
	args = parser.parse_args()

	try:
		import rasterio
		rst = rasterio.open(args.raster)
		print("CRS:", rst.crs)
		print("Transform:", rst.transform)
		w, h = rst.width, rst.height
		print("Size (W,H):", w, h)
		# 角点（像元中心）
		corners = {
			"UL": (0, 0),
			"UR": (w - 1, 0),
			"LR": (w - 1, h - 1),
			"LL": (0, h - 1),
		}
		for name, (px, py) in corners.items():
			x, y = rst.transform * (px + 0.5, py + 0.5)
			print(f"{name} center -> X:{x:.10f}, Y:{y:.10f}")
		# 像素分辨率
		print("Pixel size X:", rst.transform.a)
		print("Pixel size Y:", -rst.transform.e)
	except Exception as e:
		print("无法用rasterio读取:", e)
		print("尝试读取WorldFile:")
		base = Path(args.raster)
		wld = None
		for ext in (".tfw", ".wld", ".jgw", ".pgw"):
			cand = base.with_suffix(ext)
			if cand.exists():
				wld = cand
				break
		if not wld:
			print("未找到worldfile")
			return
		vals = [float(l.strip()) for l in Path(wld).read_text().splitlines()[:6]]
		A, D, B, E, C, F = vals
		print("WorldFile (A,D,B,E,C,F):", vals)
		print("Pixel size X:", A)
		print("Pixel size Y:", -E)
		print("Upper-left pixel CENTER X:", C)
		print("Upper-left pixel CENTER Y:", F)


if __name__ == "__main__":
	main()


