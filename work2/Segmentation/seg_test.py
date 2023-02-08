import os
import mmcv
import cv2
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, init_detector
from mmdet.apis import inference_detector, show_result_pyplot
from pycocotools.coco import COCO, maskUtils
from PIL import Image
import torch
import skimage.draw
import json
import numpy as np

cfg = Config.fromfile('/data/home/scv9609/run/mmdetection/configs/balloon/mask_rcnn.py')
config_file = '/data/home/scv9609/run/mmdetection/configs/balloon/mask_rcnn.py'
img_file = '/data/home/scv9609/run/mmdetection/data/balloon/train/14666848163_8be8e37562_k.jpg'
img_source = mmcv.imread(img_file)
checkpoint_file = '/data/home/scv9609/run/mmdetection/work/mask_rcnn/latest.pth'
score_thr=0.4
# model = build_detector(cfg.model)
model = init_detector(config_file, checkpoint_file, device='cpu')
oriimg = img_source.copy()


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def single_pic_detec(img_data,model):
    result = inference_detector(model, img_data)
    img_gray = cv2.cvtColor(img_data , cv2.COLOR_RGB2GRAY)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        # print(segms.shape)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        mask = np.zeros(img_gray.shape)
        for i in inds:
            i = int(i)
            mask = np.add(mask,segms[i])
        single_mask = mask[np.newaxis,:,:]
        single_mask = torch.tensor(single_mask > 0.5 , dtype=torch.uint8).permute(1, 2, 0).numpy()
        splash = color_splash(img_data, single_mask)
        # h, w = mask.shape[0], mask.shape[1]
        # single_mask = mask[np.newaxis,:,:]
        # single_mask = torch.tensor(single_mask > 0.5 , dtype=torch.uint8).permute(1, 2, 0).numpy()
        # img_gray = torch.tensor(img_gray).view(h, w, 1).numpy()
        # combine_output = 0.5*single_mask*img_data + 1.5*img_gray*(1 - single_mask)
        # combine_output=combine_output.astype(np.uint8)*255
        # combine_output=cv2.cvtColor(combine_output,cv2.COLOR_RGB2BGR)

    # if score_thr > 0:
    #     assert bboxes.shape[1] == 5
    #     scores = bboxes[:, -1]
    #     inds = scores > score_thr
    #     bboxes = bboxes[inds, :]

    # font_scale = 0.8
    # thickness = 3
    # bbox_color = (0, 255, 0)
    # text_color = (0, 255, 0)
    # for bbox in bboxes:
    #     bbox_int = bbox.astype(np.int32)
    #     left_top = (bbox_int[0], bbox_int[1])
    #     right_bottom = (bbox_int[2], bbox_int[3])
    #     cv2.rectangle(
    #         combine_output, left_top, right_bottom, bbox_color, thickness=thickness)
    #     if len(bbox) > 4:
    #         label_text = '{:.02f}'.format(bbox[-1])
    #     cv2.putText(combine_output, label_text, (bbox_int[0], bbox_int[1] - 5),
    #                 cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    
    return splash

def video_detec(video_in,video_out,model_file):
    # cap = cv2.VideoCapture(video_in)
    video_reader = mmcv.VideoReader(video_in)
    video_writer = None
    if video_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = single_pic_detec(frame,model)
        if video_out:
            video_writer.write(result)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    


h1, w1 = oriimg.shape[:2]
h2, w2 = oriimg.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
# new_img -=50
vis[:h1, :w1, :] = oriimg
vis[:h2, w1:w1 + w2, :] = single_pic_detec(oriimg,model)
# 保存结果
save_path = './'
out_file = os.path.join(save_path, 'result.jpg')
cv2.imwrite(out_file, vis)

video_detec('/data/home/scv9609/run/mmdetection/test_video.mp4','/data/home/scv9609/run/mmdetection/res.mp4',model)

