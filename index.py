#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import sys
sys.path.append('.')

import os
import time
from loguru import logger
import pandas as pd
import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def viss(img, boxes, scores, cls_ids, conf=0.5, class_names=None, img_id = None):
    objs = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        
        objs.append(
            [class_names[cls_id], img_id, str(float(score)), str(float(box[0])),
             str(float(box[1])), str(float(box[2])), str(float(box[3]))]
        )
    return objs

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, img_id = ''):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return []
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        objs = viss(img, bboxes, scores, cls, cls_conf, self.cls_names, img_id)
        return objs


def get_pred(exp):
    
    exp.test_conf = 0.3
    exp.nmsthre = 0.3
    model = exp.get_model()
    model.cuda()
    model.eval()
        
    ckpt_file = './best.pth'
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        'gpu', False, False,
    )

    return predictor
    
def write_csv(objs):
    logger.info(f'objs {len(objs)} x {len(objs[0])}')
    logger.info(objs)
    df = pd.DataFrame(data=objs, columns=["label", "img_id", "confidence",
                        "xmin", "ymin", "xmax", "ymax"])
    
    df.to_csv("/home/mw/project/results.csv", index=False)
    logger.info(df)
    return df

def my_image_demo(predictor, path):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    all_objs = []
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        img_id = os.path.basename(image_name)[:-4]
        objs = predictor.visual(outputs[0], img_info, predictor.confthre, img_id)
        logger.info(objs)
        if len(objs) > 0:
            for i, obj in enumerate(objs):
                all_objs.append(objs[i])
        else:
            all_objs.append(['', img_id, '','','','',''])
    
    return write_csv(all_objs)

def vid2img(file_path) :
    if not os.path.exists('./ddirs/'):
        os.makedirs('./ddirs')
    for file in os.listdir(file_path):
        path = os.path.join(file_path, file)
        if path[-4:] != '.mp4':
            continue
        logger.info(f'the path is {path}')
        vidcap = cv2.VideoCapture(path)
        logger.info(vidcap.isOpened())
        if not vidcap.isOpened():
            return False
        success,image1 = vidcap.read()
        image2 = image1
        while success:
            image2 = image1
            success, image1 = vidcap.read()
        pic_name = os.path.basename(path)[:-4] + ".jpg"
        c = os.path.join("./ddirs/", pic_name)
        cv2.imwrite(c, image2)
    return True

def invoke(_input:str) :
    res = vid2img(_input)
    if not res:
        logger.info('vid2img failed')
        return
    exp = get_exp('./exps/default/yolox_s.py')
    
    predictor = get_pred(exp)
    
    my_image_demo(predictor, "./ddirs/")

    return
    

if __name__ == "__main__":
    invoke('/gdrive/My Drive/yolo/videos')