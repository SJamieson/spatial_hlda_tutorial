import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    from notebooks.hierarchical_feature_extraction import *
except ModuleNotFoundError:
    from hierarchical_feature_extraction import *

model = models.vgg16_bn(pretrained=True).cuda()
data_dir = '/home/stewart/datasets/places365/val_256'
clusters = np.load('clusters.npy')[()]

compute_mean_diff = False

images = get_image_list(data_dir)
layers = list(clusters.keys())
random.shuffle(images)


def plot_receptive_field(cluster, image_path, idx, num_cells):
    # print(num_cells)
    img: Image.Image = Image.open(image_path).convert('RGB')
    w, h = img.size
    side_len = int(math.sqrt(num_cells))
    cell_w, cell_h = int(w / side_len), int(h / side_len)
    x, y = idx // side_len * cell_w, idx % side_len * cell_h
    x, y = y, x
    # print(x, y, side_len, cell_w, cell_h)
    plt.axis('off')
    if cell_w <= x < (w-cell_w) and cell_h <= y < (h-cell_h):
        plt.imshow(img.crop((x - cell_w, y - cell_h, x + 2 * cell_w, y + 2 * cell_h)).resize((w, h)))
    else:
        raise TypeError('bad x,y')
        plt.imshow(img.crop((x, y, x + cell_w, y + cell_h)).resize((w, h)))


mean_sum = 0
n = 0
num_test_images = 8
test_clusters = [54, 56, 65, 71, 83]
num_test_clusters = len(test_clusters)
target_layer = layers[-3]
std = 3e-10 if target_layer == 40 else 2e-6

nrow, ncol = num_test_images, num_test_clusters
with torch.no_grad():
    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    for image_idx, image_path in tqdm(enumerate(images[:num_test_images]), file=sys.stdout, total=num_test_images):
        features = {layer: [] for layer in layers}
        handles = []
        for layer in layers:
            handles.append(hook_layer(model, layer, features[layer]))
        try:
            model.forward(load_image(image_path))
        except RuntimeError:
            continue
        for layer in layers:
            num_clusters, num_dimen = tuple(clusters[layer].shape)
            deltas = np.zeros((features[layer][-1].shape[0], num_clusters))
            for i in range(num_clusters):
                deltas[:, i] = np.min(np.abs(features[layer][-1] - clusters[layer][i, :]), axis=1)
            cluster_assignments = list(np.argmin(deltas, axis=1))
            cluster_distances = np.min(deltas, axis=1)
            # print(np.average(cluster_distances))
            if layer == target_layer:
                # print('\n', np.bincount(cluster_assignments))
                # print('\n', np.bincount(cluster_assignments)[test_clusters])
                # print(len(np.bincount(cluster_assignments)))
                for cluster_num, cluster_id in enumerate(test_clusters):
                    idx = 0
                    while cluster_id in cluster_assignments[idx:]:
                        idx = cluster_assignments[idx:].index(cluster_id) + idx
                        assert cluster_assignments[idx] == cluster_id
                        if cluster_distances[idx] < std:
                            try:
                                plt.subplot(gs[image_idx, cluster_num])
                                plot_receptive_field(cluster_id, image_path, idx, features[layer][-1].shape[0])
                                break
                            except TypeError as e:
                                print('\nfailed to display image:', e)
                        idx += 1
                    else:
                        print('missing match (or all rejected)')
            if compute_mean_diff:
                mean_sum += np.mean(cluster_distances)
            n += 1
        for handle in handles:
            handle.remove()
    plt.tight_layout()
    plt.show()
if compute_mean_diff:
    print(mean_sum / n)
