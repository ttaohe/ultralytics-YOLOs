# CrossGEOView（一期：晚期融合）

跨视角晚期融合：两台手机拍摄的两张图像，先分别用YOLO推理得到检测框，然后将倾斜视角的检测通过单应性矩阵投影到“伪正射”平面，与正射图的检测在该平面做融合（WBF或NMS），可选将融合结果回投到倾斜视角，达到“被遮挡目标由另一视角补回”的效果。

## 目录结构
- `geom.py`：单应性`H(3x3)`读写、点/框透视变换、框裁剪
- `box_ops.py`：IoU、按类NMS、简化WBF
- `late_fusion.py`：核心流程（双图推理→投影→融合→回投）
- `cli.py`：命令行入口

## 依赖
- Ultralytics YOLO（本仓库）
- OpenCV（绘制与保存可视化）
- Pillow（从文件读取图像尺寸）

安装（若缺）：
```bash
pip install opencv-python pillow
```

## 单应性矩阵H格式
使用JSON文件存放3x3矩阵。
- `view->ortho`：倾斜视角到正射平面的H
- `ortho->view`：正射到倾斜视角的H（通常为前者的逆）

示例`H_view2ortho.json`：
```json
{
  "H": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
}
```

## 使用方法
从项目根目录运行：
```bash
python -m ultralytics.CrossGEOView.cli \
  /path/to/model.pt \
  /path/to/ortho.jpg \
  /path/to/view.jpg \
  /path/to/H_view2ortho.json \
  /path/to/H_ortho2view.json \
  /path/to/out_dir \
  --device cuda:0 \
  --use-wbf \
  --conf 0.25 \
  --nms-iou 0.6 \
  --wbf-iou 0.55 \
  --max-det 300
```
输出：
- `out_dir/ortho_fused.jpg`：正射平面融合可视化
- `out_dir/view_backprojected.jpg`：融合结果回投到倾斜视角的可视化
- `out_dir/ortho_fused.json`：融合结果（boxes/scores/classes）

## 生成伪地理坐标（GeoTIFF 或 WorldFile）
若你有俯视/伪正射影像，但尚无地理参照，可用：
```bash
python -m ultralytics.CrossGEOView.pseudo_georef \
  /path/to/ortho.jpg \
  <lat_min> <lon_min> \
  <meters_per_pixel> \
  /path/to/ortho_georef.tif \
  --lat-mid <optional_lat_ref>
```
说明：
- 输入左下角经纬度与像素分辨率（米/像素），脚本会近似换算为度/像素并写入WGS84。
- 若安装`rasterio`则输出GeoTIFF；否则在源图同名位置写`.wld`和`.prj`（WGS84），原图不改名。
- 近似换算在小范围内适用，若需严格米制请改用投影坐标（UTM）后再转换。

## 备注与建议
- 若只想快速验证融合效果，可只提供`H_view2ortho.json`并查看`ortho_fused.jpg`；回投可选。
- `--use-wbf`在目标密集区域通常优于NMS；若配准误差较大可适当调大`--wbf-iou`或改用NMS。
- 正射平面分辨率应与`ortho.jpg`一致；确保H的定义与该尺寸匹配。

## 基于H与GeoTIFF的地理位置编码（经纬度网格）
为任一视角图像生成经纬度编码（下采样以提速）并可视化：
```bash
python -m ultralytics.CrossGEOView.encode_pos \
  /path/to/view.jpg \
  /path/to/H_view2ortho.json \
  /path/to/ortho_georef.tif \
  /path/to/out_dir \
  --stride 4
```
输出：
- `out_dir/lon.npy`, `lat.npy`, `mask.npy`：经纬度网格与有效掩码
- `out_dir/geope.png`：位置编码可视化（按局部min-max归一化后的热力图）
