import io
import torch
# from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Blueprint
from torchvision import models, transforms

from model import NIMA
from demo import bp
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# from scipy.misc import imread
# %matplotlib inline

#app = Flask(__name__)
app = Blueprint('pytorch', __name__)
#app.register_blueprint(bp)

#device = torch.device("cpu")
#base_model = models.vgg16(pretrained=True)
base_model = models.vgg16()
model = NIMA(base_model)
model.load_state_dict(torch.load("/golem/entrypoints/aethestics/src/weights/nima.pth",map_location='cpu'))
#model.load_state_dict(torch.load("weights/nima.pth"))
#model = model.to(device)
model.eval()



val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_score(img):
    #new_img = val_transform(img).unsqueeze(0).to(device)
    new_img = val_transform(img).unsqueeze(0)
    #outputs = model(new_img)
    outputs = model(torch.autograd.Variable(new_img))
    #dist = torch.arange(1, 11).float().to(device)
    dist = torch.arange(1, 11).float()
    outputs = outputs.data
    p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
    return float(p_mean)
    #return float(torch.mean(outputs)

def get_dist(img):
    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    new_img = val_transform(img).unsqueeze(0).to(device)
    outputs = model(new_img)
    dist = torch.arange(1, 11).float().to(device)
    p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
    np_dist = outputs.data.cpu().numpy()
    result = {'dist': np_dist.tolist(), 'var': np.var(np_dist).tolist()}
    return jsonify(result)


def remap(x):
    return sigmoid(x - 4.5) * 10


file_name = "/golem/resource/demo.jpg"
# imshow(imread(file_name, 1))

img = Image.open(file_name).convert("RGB")

remap_score = remap(get_score(img))
print("{:.2f}".format(remap_score))