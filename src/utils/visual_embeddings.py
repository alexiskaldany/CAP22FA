from transformers import BertTokenizer,DeiTFeatureExtractor,VisualBertForMultipleChoice
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def copy_embeddings(m, i, o):
        """Copy embeddings from the penultimate layer.
        """
        o = o[:, :, 0, 0].detach().numpy().tolist()
        embeddings.append(o[0][0])
        return 
def load_image_resize_convert(image_path):
    image = Image.open(image_path)
    image = image.resize(size=(480, 620), resample=Image.BILINEAR)
    image = image.convert("RGB")
    image = transforms.ToTensor()(image)
    return image

def get_multiple_embeddings(list_of_images:list):
    embeddings = []
    image_embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    layer = image_embedding_model._modules.get('avgpool')
    _ = layer.register_forward_hook(copy_embeddings)
    image_embedding_model.eval() 
    for image in list_of_images:
        image_tensor = load_image_resize_convert(image)
        image_tensor = image_tensor.unsqueeze(0)
        _ = image_embedding_model(image_tensor)
    return embeddings