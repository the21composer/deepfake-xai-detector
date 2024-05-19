# %%
# Loading the required libraries
import os, sys, time
import cv2
import random
import numpy as np
import pandas as pd
import skimage.transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from torch import topk

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow
from tqdm.notebook import tqdm


# %%
# Read the test videos 
test_dir = "./data"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)

test_labels = pd.read_csv('./data/labels.csv')
test_labels.head()

# %%
test_labels['label'].value_counts()

# %%
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

# %%
# Check if GPU is available
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu

# %%
# Attach the required libraries to the system path
# This have a few helper functions & path to the pre-trained model

import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")

# %%
# Initalize blazeface 

from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)

# %%
from read_video_1 import VideoReader
from face_extract_1 import FaceExtractor

frames_per_video = 16

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

# %%
input_size = 224 # Define the input size of the image

# Define the normalizing functions with ImageNet parameters 
from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

# Define some helper functions for re-sizing image & making them into perfect squares
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

# %%
import torch.nn as nn
import torchvision.models as models

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        
        self.fc = nn.Linear(2048, 1)

# %%
#Load the checkpoint & update the model for prediction
checkpoint = torch.load("./resnext.pth", map_location=gpu)

model = MyResNeXt().to(gpu)
model.load_state_dict(checkpoint)
_ = model.eval()

del checkpoint

# %%
y = 'REAL'
while y == 'REAL':
    sample_video= random.choice(test_videos) # Select a random test video 
    #coadfnerlk.mp4
    #zgbhzkditd.mp4
    #bwdmzwhdnw.mp4
    #sample_video = 'zgbhzkditd.mp4'
    video_path = os.path.join(test_dir, sample_video)
    #y = test_labels[test_labels['processedVideo'] == sample_video]['label'].values[0]
    y = 'FAKE'
print("Selected Video: ", sample_video)
#print("True Value: ",y)
#zcxcmneefk.mp4
#lmdyicksrv.mp4
#jhczqfefgw.mp4


# %%
batch_size = 16 # Extract faces from 16 frames in the video
faces = face_extractor.process_video(video_path)
print("No. of frames extracted: ", len(faces))
print("Keys in the extracted info: ", faces[0].keys())
try:
    print("Shape of extracted face_crop: ", faces[0]['faces'][0].shape) # multiple faces can be captured. In this set only a single face is detected
    print("Scores of the face crop: ", faces[0]['scores'][0])
except:
    print("=====================================")
    print("No faces detected! Please run again.")

# %%
# Only look at one face per frame. This removes multiple faces from each frame, keeping only the best face
face_extractor.keep_only_best_face(faces)

# %%
print(len(faces))

# %%
sample_face = faces[1]['faces'][0]
resized_face = isotropically_resize_image(sample_face, input_size)
resized_face = make_square_image(resized_face)
resized_face.shape

# %%
x = torch.tensor(resized_face, device=gpu).float()
print(x.shape)
# Preprocess the images.
x = x.permute(( 2, 0, 1))
x = normalize_transform(x / 255.)
x = x.unsqueeze(0)
print(x.shape)

# %%
prediction_var = Variable(x.cuda(), requires_grad=True) # Squeeze the  variable to add an additional dimension & then
prediction_var.shape                                    # wrap it in a Variable which stores the grad_training weights

# %%
y_pred = model(prediction_var)
y_pred = torch.sigmoid(y_pred.squeeze())

print("Prediction: ", y_pred)
pred_probabilities = F.softmax(y_pred).data.squeeze() # Pass the predictions through a softmax layer to convert into probabilities for each class
print("Predicted Class: ", pred_probabilities)

# %%
from pytorch_gradcam import HiResCAM, GradCAM
from pytorch_gradcam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_gradcam.utils.image import show_cam_on_image

def repeat_matrix(matrix, N):
    m, n = matrix.shape
    new_shape = (m * N, n * N)
    repeated_matrix = np.empty(new_shape)

    for i in range(m * N):
        for j in range(n * N):
            repeated_matrix[i, j] = 1 / (1 + np.exp(10 * (-(matrix[i // N, j // N] - 0.5))))

    return repeated_matrix

def initCAM(model, target_layers):
    cam_last = HiResCAM(model=model, target_layers=target_layers[0], use_cuda=True)
    cam_prelast = HiResCAM(model=model, target_layers=target_layers[1], use_cuda=True)
    return [cam_last, cam_prelast]

def getHiResCAM(data, cams, targets):
    grayscale_cam_last = cams[0](input_tensor=data, targets=targets)
    grayscale_cam_last = grayscale_cam_last[0, :]
    grayscale_cam_prelast = cams[1](input_tensor=data, targets=targets)
    grayscale_cam_prelast = grayscale_cam_prelast[0, :]
    output = repeat_matrix(grayscale_cam_last, 1) * grayscale_cam_prelast
    return grayscale_cam_last, grayscale_cam_prelast, output

cams = initCAM(model, [[model.layer4[-1]], [model.layer3[-1]]])

#imshow(visualization)

# %%
target = [BinaryClassifierOutputTarget(1)]
def explainDeepFake(samples):
    output = []
    for sample in samples:
        if (len(sample['faces'])):
            sample_face = sample['faces'][0]
            resized_face = isotropically_resize_image(sample_face, input_size)
            resized_face = make_square_image(resized_face)
            resized_face.shape
            x = torch.tensor(resized_face, device=gpu).float()
            x = x.permute(( 2, 0, 1))
            x = normalize_transform(x / 255.)
            x = x.unsqueeze(0)
            prediction_var = Variable(x.cuda(), requires_grad=True)
            grayscale_cam_last, grayscale_cam_prelast, hirescamplus_cam = getHiResCAM(prediction_var, cams, target)
            resized_face_norm = resized_face / 255
            output.append([grayscale_cam_last, grayscale_cam_prelast, hirescamplus_cam, resized_face_norm])
    return output

output = explainDeepFake(faces)
fig, ax = plt.subplots(16,4, figsize=(30,160))
i = 0
for explain in output:
    visualization_last = show_cam_on_image(explain[3], explain[0], use_rgb=True)
    ax[i, 0].imshow(visualization_last)
    visualization_last = show_cam_on_image(explain[3], explain[1], use_rgb=True)
    ax[i, 1].imshow(visualization_last)
    visualization_last = show_cam_on_image(explain[3], explain[2], use_rgb=True)
    ax[i, 2].imshow(visualization_last)
    ax[i, 3].imshow(explain[3])
    i += 1
        