import matplotlib.pyplot as plt
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import copy
import matplotlib.patheffects as PathEffects
import random
#from data import SegData, SynData
import os
from glob import glob
from pathlib import Path

from torchvision import transforms
import cv2 as cv


# with open("../references/segmentation-deloitte-colour-map.json", mode = 'r', encoding='utf-8') as f:
#     colour_map = json.load(f)

#colour_map = {int(k):v for k,v in colour_map.items()}

# def map_indices_to_rgb(tensor_2d, mapping_dict = colour_map):
#     """
#     Map each cell in a 2D tensor to an RGB value based on a provided dictionary.

#     :param tensor_2d: A 2D tensor with indices.
#     :param mapping_dict: A dictionary mapping indices to RGB values.
#     :return: A 3D tensor with RGB values.
#     """
#     # Create a 3D tensor to hold the RGB values
#     rgb_tensor = torch.zeros(tensor_2d.shape[0], tensor_2d.shape[1], 3, dtype=torch.uint8)

#     # Iterate over the 2D tensor and map each value
#     for x in range(tensor_2d.shape[0]):
#         for y in range(tensor_2d.shape[1]):
#             index = tensor_2d[x, y].item()
#             if index in mapping_dict:
#                 rgb_tensor[x, y] = torch.tensor(mapping_dict[index]["color values"], dtype=torch.uint8)

#     return rgb_tensor
# def find_active_layers_all(tensor_3d):
#     """
#     Returns a 2D tensor representing the active layer indices for each pixel position in the 3D tensor.

#     :param tensor_3d: A 3D PyTorch tensor of shape (10, 256, 256).
#     :return: A 2D tensor of shape (256, 256) with the indices of the active layers.
#              If no layer is active at a position, the value will be -1.
#     """
#     # Find indices where value is 1 in each layer
#     active_indices = tensor_3d == 1.

#     # Create a tensor to hold the result
#     result = torch.full((256, 256), -1, dtype=torch.long)

#     # Loop over each layer
#     for i in range(tensor_3d.shape[0]):
#         # Update the result tensor with the layer index where the condition is met
#         result[active_indices[i]] = i

#     return result


# def mask2rgb(mask):
#     """Converts a segmentation mask (C,H,W) to RGB (H,W,3)"""
#     index_mask = find_active_layers_all(mask)*10
#     rgb = map_indices_to_rgb(index_mask)
#     return rgb


# def showSegmentation(img, mask, pred=None, title=None):
#     """Shows a segmentation mask overlaid on an image"""
#     if img.shape == (3,256,256):
#         img = img.permute(1,2,0)
#         img = img/255 # Convert to RGB
    
#     mask = mask2rgb(mask)
#     mask = mask # Convert to RGB
#     if pred is None:
#         plt.imshow(img)
#         plt.imshow(mask, alpha=0.5)
#         plt.title("Image with hand labelled mask")
#     else:
#         pred = map_indices_to_rgb(pred)
#         fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#         # Plot the handlabelled image
#         ax[0].imshow(img)
#         ax[0].imshow(mask, alpha=0.5)
#         ax[0].set_title("Image with hand labelled mask")
#         # Plot the predicted segmentation mask
#         ax[1].imshow(img)
#         ax[1].imshow(pred, alpha=0.5)
#         ax[1].set_title("Image with predicted mask")




def plot_eval(img, mask, pred, classes, iou, norm=True):
    name = "test_plots/" + str(random.randint(0, 100000))+".png"
    my_cmap = copy.copy(cm.jet)
    my_cmap.set_under('k', alpha=0)
    
    if norm:
        stds = torch.tensor([0.229, 0.224, 0.225])
        means = torch.tensor([0.485, 0.456, 0.406])
        img *= stds[:, None, None]
        img += means[:, None, None]
    
    img = (img - img.min()) / (img.max() - img.min())
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img.permute(1,2,0))
    ax[0].axis("off")
    ax[0].set_title("Image")
    
    ax[1].imshow(img.permute(1,2,0))
    ax[1].imshow(mask, alpha=0.7, cmap=my_cmap, interpolation='none', clim=[1, classes]) 
    ax[1].axis("off")
    ax[1].set_title("Ground Truth")
    
    ax[2].imshow(img.permute(1,2,0))
    ax[2].imshow(pred,alpha=0.7,cmap=my_cmap, interpolation='none', clim=[1, classes])
    ax[2].axis("off")
    ax[2].set_title("Prediction")

    fig.suptitle("mIoU: " + str(round(np.nanmean(iou[1:]),2)))
    
    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    return name
