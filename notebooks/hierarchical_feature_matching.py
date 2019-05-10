import numpy as np
from torchvision import models
import timeit
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    from notebooks.hierarchical_feature_extraction import *
except ModuleNotFoundError:
    from hierarchical_feature_extraction import *

model = models.vgg16_bn(pretrained=True).cuda()
data_dir = '/home/stewart/datasets/places365/val_256'
feature_dir = 'output/'
clusters = np.load(feature_dir + 'clusters.npy')[()]
compute_mean_diff = True

images = get_image_list(data_dir)
random.shuffle(images)

layers = list(clusters.keys())


def get_receptive_field(cluster, image_path, idx, num_cells, show=False):
    img: Image.Image = Image.open(image_path).convert('RGB')
    w, h = img.size
    side_len = int(math.sqrt(num_cells))
    cell_w, cell_h = int(w / side_len), int(h / side_len)
    x, y = idx // side_len * cell_w, idx % side_len * cell_h
    plt.axis('off')
    if cell_w <= x < (w - cell_w) and cell_h <= y < (h - cell_h):
        img = img.crop((x - cell_w, y - cell_h, x + 2 * cell_w, y + 2 * cell_h)).resize((w, h))
    else:
        img = img.crop((x, y, x + cell_w, y + cell_h)).resize((w, h))
    if show:
        plt.imshow(img)
    return img


mean_sum = 0
mean_std = 0
n = 0
num_images = 1000
num_exemplars = 8
# test_clusters = [54, 56, 65, 71, 83]
test_clusters = list(range(1, 7))
num_test_clusters = len(test_clusters)

exemplars = {cluster: [] for cluster in test_clusters}

for target_layer in layers:
    with open(feature_dir + f'kmeans-{target_layer}.pkl', 'rb') as f:
        kmeans = pickle.load(f) # type: cluster.MiniBatchKMeans

    with torch.no_grad():
        layer = target_layer
        for image_path in tqdm(images[:num_images], file=sys.stdout):
            features = {layer: [] for layer in layers}
            handle = hook_layer(model, layer, features[layer])
            try:
                model.forward(load_image(image_path))
            except RuntimeError:
                continue
            deltas = kmeans.transform(features[layer][-1])
            cluster_assignments = list(np.argmin(deltas, axis=1))
            cluster_distances = np.min(deltas, axis=1)
            for cluster_num, cluster_id in enumerate(test_clusters):
                idx = 0
                while cluster_id in cluster_assignments[idx:]:
                    idx = cluster_assignments[idx:].index(cluster_id) + idx
                    # assert cluster_assignments[idx] == cluster_id
                    if len(exemplars[cluster_id]) < num_exemplars or cluster_distances[idx] < exemplars[cluster_id][-1][
                        1]:
                        try:
                            img = get_receptive_field(cluster_id, image_path, idx, features[layer][-1].shape[0])
                            if len(exemplars[cluster_id]) < num_exemplars:
                                exemplars[cluster_id].append((img, cluster_distances[idx]))
                            else:
                                exemplars[cluster_id][-1] = (img, cluster_distances[idx])
                        except TypeError as e:
                            print('\nfailed to display image:', e)
                    idx += 1
                exemplars[cluster_id].sort(key=lambda x: x[1])  # We wait until now to sort so we have <=1 example/image
            if compute_mean_diff:
                mean_sum += np.mean(cluster_distances)
                mean_std += np.std(cluster_distances)
            n += 1
            handle.remove()
    if compute_mean_diff:
        print(mean_sum / n)
        print(mean_std / n)
    for cluster in test_clusters:
        print([e[1] for e in exemplars[cluster]])

    nrow, ncol = num_exemplars, num_test_clusters
    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    for col, cluster in enumerate(test_clusters):
        for row in range(len(exemplars[cluster])):
            plt.subplot(gs[row, col])
            if row == 0:
                plt.title(f'{target_layer}-{cluster}')
            plt.axis('off')
            plt.imshow(exemplars[cluster][row][0])
    plt.tight_layout()
    # plt.title(f'Exemplars from layer {target_layer}')
    plt.savefig(f'figures/exemplars-{target_layer}.png')
    plt.show()
