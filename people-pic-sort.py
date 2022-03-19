import os
import pathlib
import shutil
import cv2
import PIL
import numpy as np

import torch
from torchvision import models, transforms

# CONFIG
source_folder = "/data/cloud/OneDrive/Pictures"
target_folder_person = "/data/pics/person"
target_folder_noperson = "/data/pics/no-person"
person_threshold = 0.45
detector_model = models.detection.retinanet_resnet50_fpn
image_res = (512, 512)

# Function to iterate through all folders and sub folderse for Images
def get_all_files(folder_path):
    file_list =[]
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isdir(item_path):
            file_list += get_all_files(folder_path=item_path)
        elif os.path.isfile(item_path):
            file_list.append(item_path)
    return file_list

# Setup Pytorch
model_device_name = "cpu"
if torch.cuda.is_available():
    model_device_name = "cuda"
print ("Pytorch Configured for device:", model_device_name)
model_device = torch.device(model_device_name)
model =  detector_model(pretrained=True, progress=True, pretrained_backbone=True).to(model_device)
model.eval()

# Search all files and create list of images to filter 
print ("Searching all files...")
all_files = get_all_files(source_folder)
print ("Found: ", len(all_files))

# Setup  OpenCV windows for showing preson and non-person images
cv2.namedWindow('person',cv2.WINDOW_NORMAL)
cv2.namedWindow('no_person',cv2.WINDOW_NORMAL)

# Iterate through all images for .jpg files
print ("Sorting...")
for file_path in all_files:
    file_ext = str.upper(pathlib.Path(file_path).suffix)
    if file_ext  == ".JPG":
        
        # read image & resize
        pil_image = PIL.Image.open(file_path)
        pil_image = pil_image.resize(image_res)
        
        # Convert to Tensor & Add batch dimension
        transform = transforms.ToTensor()
        tensor_img = transform(pil_image).unsqueeze(0).to(model_device)
               
        # Run Model
        detections = model(tensor_img)[0]        

        # create openCV image for display
        img_cv = np.array(pil_image) 
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # loop over the detections
        person_found = False
        for i in range(0, len(detections["boxes"])):
            
            confidence = detections["scores"][i]
            idx = int(detections["labels"][i])
            if confidence > person_threshold and idx == 1:
                person_found = True
                
                # Get bbox dimenstions                
                bbox = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = bbox.astype("int")
                
                # draw the bounding box and label on the image
                label = "{:.2f}%".format(confidence * 100)
                cv2.rectangle(img_cv, (startX, startY), (endX, endY), (255,0,255), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img_cv, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        # Copy files & display the output
        if person_found:
            shutil.copy(file_path, target_folder_person + "/")
            cv2.imshow('person', img_cv)
        else:
            shutil.copy(file_path, target_folder_noperson + "/")
            cv2.imshow('no_person', img_cv)

        # Exit on ESCAPE
        x = cv2.waitKey(1)
        if x == 27:
            exit()

print ("Done!")



