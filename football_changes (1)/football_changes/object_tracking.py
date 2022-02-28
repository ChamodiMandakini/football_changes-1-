# importing the correct libraries
from object_tracker.models import Darknet
from object_tracker.utils import load_classes, non_max_suppression
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageColor
from object_tracker.color_selector import colorSelector
import cv2 as cv
from object_tracker.sort import Sort
from os import getcwd
from os.path import join

# path configuration
config_path = join('object_tracker', 'config', 'yolov3.cfg')
weights_path = join('object_tracker', 'config', 'yolov3.weights')
class_path = join('object_tracker', 'config', 'coco.names')
img_size = 416
conf_thres = 0.8
nms_thres = 0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to("cpu")
model.eval()
classes = load_classes(class_path)
Tensor = torch.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),\
            transforms.Pad((max(int((imh-imw)/2), 0),\
                max(int((imw-imh)/2), 0),\
            max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0)),\
            (128, 128, 128)), transforms.ToTensor(), ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def process_image(frame, mot_tracker, color1, color2, color3):
    """Process each frame\
            Identifies objects in the image and sort out the "players" into
            teams according
            to the color of their teams.
    arguments:
        frame -> each frame of the video
        mot_tracker -> Sort()
        color1 -> color of the first team
        color2 -> color of the second team
        color3 -> color of the third team
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            # appending my data here
            # have a better explanination here
            bbox = tuple(map(lambda x: 0 if x < 0 else x,
                (int(x1),int(y1),int(box_w),int(box_h))))
            # getting the cropped image
            thisimg = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[3]]
            # replace the array with wanted colors
            # if the predicted image is a player configure the team
            if classes[int(cls_pred)] == 'player':
                color,cls = colorSelector(thisimg, [color1, color2, color3])
                cv.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                cv.putText(frame, cls, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            else:
                continue
    return cv.cvtColor(frame,cv.COLOR_BGR2RGB)

def extract_images(filename,filename_ext, start=-1, end=-1, color1=np.array([60, 60, 100]),\
        color2=np.array([240, 100, 100]), color3=np.array([0, 100, 100])):
    """Extract the Images in a Video
    arguments:
        filename -> video filename
        filename_ext -> filename without the extension
        start -> start frame
        end -> end frame
        color1 -> color of the first team
        color2 -> color of the second team
        color3 -> color of the third team
    """
    cap = cv.VideoCapture(filename)
    
    print(f"Three colors are\n\t{color1}\n\t{color2}\n\t{color3}")
    if start < 0:
        start = 0
    if end < 0:
        end = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    cap.set(1, start)
    frame = start
    
    while frame < end:
        ret, image = cap.read()
        if image is None:
            continue
        filename = f"img_{frame}.jpg"
        filepath = join(getcwd(),'static', 'images', filename_ext, filename)
        print(f"inside obj_track: image is saved at {filepath}")
        frame += 1
        cv.imwrite(filepath, process_image(image, Sort(), color1, color2,
            color3))
    cap.release()
    return frame
