"""
visual_embeddings.py
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""
from src.utils.configs import IMAGE_DIMENSIONS
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from loguru import logger
from pathlib import Path
from torchvision.models.resnet import resnet18 as _resnet18

def copy_embeddings(m, i, o):
        """Copy embeddings from the penultimate layer.
        """
        o = o[:, :, 0, 0].detach().numpy().tolist()
        embeddings.append(o)
        return 

def load_image_resize_convert(image_path):
    preprocess = transforms.Compose([
    transforms.ToTensor(),transforms.Resize(site=(224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = preprocess(Image.open(image_path))
    input_batch = input_tensor.unsqueeze(0)
    print(input_tensor.shape)
    print(input_batch.shape)
    return input_batch

def get_multiple_embeddings(list_of_images:list):
    global embeddings
    embeddings = []
    id_embedding_dict = {}
    # image_embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=models.ResNet18_Weights.DEFAULT')
    image_embedding_model = _resnet18(weights='DEFAULT')
    layer = image_embedding_model._modules.get('avgpool')
    _ = layer.register_forward_hook(copy_embeddings)
    image_embedding_model.eval() 
    for idx,image in enumerate(set(list_of_images)):
        if idx % 100 == 0:
            logger.info(f"At index:{idx}")
        image_tensor = load_image_resize_convert(image)
        _ = image_embedding_model(image_tensor)
        id_embedding_dict[Path(image).name.split(".")[0]] = embeddings.pop()[0]
        #image.split("/")[-1].split(".")[0]
    return id_embedding_dict