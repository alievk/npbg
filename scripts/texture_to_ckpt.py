import cv2
import argparse
from pathlib import Path
import sys, os
import torch

from npbg.models.texture import MeshTexture
from npbg.models.compose import RGBTexture

parser = argparse.ArgumentParser()
parser.add_argument('texture', type=str)
args = parser.parse_args()

image = cv2.imread(args.texture)[::-1, :, ::-1].copy()

tex = MeshTexture(3, image.shape[0], activation='none', levels=1, reg_weight=0)
tex.texture_0 = torch.nn.Parameter(torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255)

model = RGBTexture(tex)

ckpt = Path(args.texture).parent / 'model.pth'

model_args = {
    'pipeline': 'npbg.pipelines.ogl.RGBTexturePipeline', 
    'texture_size': image.shape[0],
    'input_format': 'uv_2d'
    }

torch.save({'args': model_args, 'state_dict': model.state_dict()}, ckpt)