from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
import torch
import os
import mmcv
import cv2

config_file = '/data/home/scv9609/run/mmdetection/configs/balloon/ssd.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/data/home/scv9609/run/mmdetection/work/ssd/latest.pth'

# test a single image
img = '/data/home/scv9609/run/mmdetection/demo/demo.jpg'
img_out_dir = '/data/home/scv9609/run/mmdetection/demo/demo_res.jpg'
# /data/home/scv9609/run/mmdetection/data/VOCdevkit/VOC2007/JPEGImages/002971.jpg
# result = inference_detector(model, img)

# init a detector
model = init_detector(config_file, checkpoint_file, device='cpu')
# inference the demo image
 
# for filename in os.listdir(img):
# img = os.path.join(imagepath, filename)
result = inference_detector(model, img)
# out_file = os.path.join(savepath, filename)
show_result_pyplot(model, img, result,score_thr=0.6,out_file =img_out_dir)
