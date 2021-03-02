# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torchvision import transforms
import json
from models import *
from azureml.core.model import Model

from utils.datasets import *
from utils.utils import *


import base64

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best.pt')
    print(model_path)
    device = torch.device('cpu')

    # Initialize model
    #model = Darknet(opt.cfg, img_size)
    cfg = 'Detect-Yolov3-Sample/cfg/yolov3-tiny-3cls.cfg'
    print(os.listdir('.'))
    model = Darknet(cfg,512)
    
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.to(device).eval()


def run(input_data):
    img_size = 512
    device = torch.device('cpu')
    conf_thres = 0.3
    iou_thres = 0.6
    half = False

    #input_data = torch.tensor(json.loads(input_data)['data'])
    raw_data = json.loads(input_data)["data"]
    image_data = base64.b64decode(raw_data)
    with open('Detect-Yolov3-Sample/test-image.jpg', 'wb') as jpg_file:
        jpg_file.write(image_data)
    
    #img = input_data
    dataset = LoadImages('Detect-Yolov3-Sample/test-image.jpg', img_size=img_size)
    
    # Get names and colors
    names = load_classes('Detect-Yolov3-Sample/data/detect.names')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
    #    t1 = torch_utils.time_synchronized()
    pred = model(img, augment='false')[0]
    #    t2 = torch_utils.time_synchronized()

    pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes='', agnostic='aflse')

    objects = ''
    objectid = 1
    p, s, im0 = path, '', im0s
    
    for i, det in enumerate(pred):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            tmp = { 
                'id': str(objectid),
                'names':names[int(cls)],
                'probability': round(float(conf),3),
                'position': (int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))
            }
            if objectid == 1:
                objects = str(tmp)
            else:
                objects = objects +', '+str(tmp)
            objectid = objectid + 1

            #write image with detections
            label = '%s %.2f' % (names[int(cls)], conf)
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

    cv2.imwrite('Detect-Yolov3-Sample/test-image-detections.jpg', im0)

    with open('Detect-Yolov3-Sample/test-image-detections.jpg', 'rb') as jpg_file:
        byte_content = jpg_file.read()

    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode() 

    result = {'objects': [ objects ],'object_count': str(objectid - 1),'result_image': base64_string}
    
    result = json.dumps(result)
    
    return result
