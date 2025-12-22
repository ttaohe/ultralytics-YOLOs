WEIGHT=/home/hetao/graduate/ultralytics-YOLOs/runs/train/train_yolov10l_visroneDet_epoch300_imgsz1280_batch8_mixup02/weights/best.pt

yolo val split=test data=VisDrone.yaml imgsz=1280 batch=1 model=$WEIGHT name=test_yolo10l_mixup02_epoch300_visdronedet