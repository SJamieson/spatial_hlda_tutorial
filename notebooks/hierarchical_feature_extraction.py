import os
import sys
import math
import random
import torch
import torch.nn
import numpy as np
import pickle
from sklearn import cluster
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
from PIL import Image

def hook_layer(model, idx, features, debug=False):
    def get_hook(features):
        def hook(module, input, output):
            data: np.ndarray = output.cpu().numpy()
            features.append(data.reshape((-1, np.prod(data.shape[2:]))).transpose((1, 0)))
            if debug:
                assert np.equal(features[-1][0,:], data[0,:,0,0]).all()
                assert np.equal(features[-1][1,:], data[0,:,0,1]).all()
                assert np.equal(features[-1][data.shape[-1],:], data[0,:,1,0]).all()
        return hook
    layer : torch.nn.Module = model._modules['features'][idx]
    return layer.register_forward_hook(get_hook(features))

def load_image(path, gpu=False):
    img = Image.open(path)
    img_tensor : torch.Tensor = transforms.ToTensor()(img)
    if gpu:
        img_tensor = img_tensor.cuda()
    return img_tensor.reshape(1, *img_tensor.shape)

def get_image_list(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if '.jpg' in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return images

def get_batch(i, batch_size):
    with torch.no_grad():
        for i in range(i*batch_size, min(len(images), (i+1)*batch_size)):
            try:
                model.forward(load_image(images[i], gpu=True))
            except RuntimeError as e:
                print('Skipping', images[i])
    # return 1
    return math.ceil(len(images) / float(batch_size))


if __name__ == "__main__":
    model = models.vgg16_bn(pretrained=True)
    data_dir = '/home/stewart/datasets/places365/val_256'
    layers = [20, 30, 40]
    num_clusters = [800, 400, 200]
    num_epochs = 2
    batch_size = 64

    model.cuda()
    images = get_image_list(data_dir)
    random.shuffle(images)

    clusters = {}
    for i, layer_idx in enumerate(layers):
        epoch = 0
        kmeans = cluster.MiniBatchKMeans(n_clusters=num_clusters[i], max_iter=300, batch_size=batch_size)
        batch_idx = 0
        counter = tqdm(file=sys.stdout)
        while epoch < num_epochs:
            features = []
            hook = hook_layer(model, layer_idx, features)
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
        with open(f'kmeans-{layer_idx}.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        # clusters[layer_idx] = cluster.k_means(features[layer_idx],
        #                                       n_clusters=num_clusters[i],
        #                                       precompute_distances=True,
        #                                       verbose=True)

    np.save('clusters.npy', clusters)