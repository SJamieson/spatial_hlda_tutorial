import os
import sys
import math
import random
import torch
import torch.nn
import numpy as np
from sklearn import cluster
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
from PIL import Image

model = models.vgg16_bn(pretrained=True)
data_dir = '/home/stewart/datasets/places365/val_256'
layers = [20, 30, 40]
num_clusters = [1000, 200, 100]
num_epochs = 1
batch_size = 32

def get_hook(idx, features):
    def hook(module, input, output):
        data : np.ndarray = output.cpu().numpy()
        features.append(data.reshape((-1, np.prod(data.shape[2:]))).transpose((1, 0)))
        # assert np.equal(features[idx][-1][0,:], data[0,:,0,0]).all()
    return hook

def hook_layer(idx, features):
    layer : torch.nn.Module = model._modules['features'][idx]
    return layer.register_forward_hook(get_hook(idx, features))

def load_image(path):
    img = Image.open(path)
    img_tensor : torch.Tensor = transforms.ToTensor()(img)
    return img_tensor.reshape(1, *img_tensor.shape).cuda()

model.cuda()
images = []
for root, _, fnames in sorted(os.walk(data_dir)):
    for fname in sorted(fnames):
        if '.jpg' in fname:
            path = os.path.join(root, fname)
            images.append(path)
random.shuffle(images)

def get_batch(i, batch_size):
    with torch.no_grad():
        for i in range(i*batch_size, min(len(images), (i+1)*batch_size)):
            try:
                model.forward(load_image(images[i]))
            except RuntimeError as e:
                print('Skipping', images[i])
    return math.ceil(len(images) / float(batch_size))

clusters = {}
for i, layer_idx in enumerate(layers):
    epoch = 0
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_clusters[i], max_iter=300, batch_size=batch_size)
    batch_idx = 0
    counter = tqdm(file=sys.stdout)
    while epoch < num_epochs:
        features = []
        hook = hook_layer(layer_idx, features)
        num_batches = get_batch(batch_idx, batch_size=batch_size)
        counter.total = num_batches * num_epochs
        features = np.concatenate(features)
        hook.remove()
        kmeans.partial_fit(features)
        counter.update()
        batch_idx += 1
        if batch_idx >= num_batches:
            batch_idx = 0
            epoch += 1
            print('Epoch', epoch)
    clusters[layer_idx] = kmeans.cluster_centers_
    # clusters[layer_idx] = cluster.k_means(features[layer_idx],
    #                                       n_clusters=num_clusters[i],
    #                                       precompute_distances=True,
    #                                       verbose=True)

np.save('clusters.npy', clusters)