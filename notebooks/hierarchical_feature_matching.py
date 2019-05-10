from typing import List
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    from notebooks.hierarchical_feature_extraction import *
except ModuleNotFoundError:
    from hierarchical_feature_extraction import *

data_dir = '/home/stewart/datasets/places365/val_256'
feature_dir = 'output/'
clusters = np.load(feature_dir + 'clusters.npy')[()]
compute_mean_diff = True
vocab_sizes = [cluster.shape[0] for cluster in clusters.values()]

images = get_image_list(data_dir)
# random.shuffle(images)

layers = list(clusters.keys())


def get_receptive_field(cluster, image_path, idx, num_cells, show=False):
    img: Image.Image = Image.open(image_path).convert('RGB')
    w, h = img.size
    side_len = int(math.sqrt(num_cells))
    cell_w, cell_h = int(w / side_len), int(h / side_len)
    x, y = idx // side_len * cell_w, idx % side_len * cell_h
    plt.axis('off')
    if cell_w * 2 <= x < (w - cell_w * 3) and cell_h * 2 <= y < (y - cell_h * 3):
        img = img.crop((x - cell_w*2, y - cell_h*2, x + 3 * cell_w, y + 3 * cell_h)).resize((w, h))
    elif cell_w <= x < (w - cell_w * 2) and cell_h <= y < (h - cell_h * 2):
        img = img.crop((x - cell_w, y - cell_h, x + 2 * cell_w, y + 2 * cell_h)).resize((w, h))
    else:
        img = img.crop((x, y, x + cell_w, y + cell_h)).resize((w, h))
    if show:
        plt.imshow(img)
    return img


mean_sum = 0
mean_std = 0
n = 0
num_validation_images = 500
num_test_images = 1000
num_exemplars = 7
# test_clusters = [54, 56, 65, 71, 83]
# test_clusters = list(range(1, 7))
num_test_clusters = 6

image_words = {image_path: {layer: [] for layer in layers} for image_path in images[:num_test_images]}

for target_layer in layers:
    test_clusters = []
    with open(feature_dir + f'kmeans-{target_layer}.pkl', 'rb') as f:
        kmeans = pickle.load(f)  # type: cluster.MiniBatchKMeans

    layer = target_layer
    cluster_scores = None
    model = models.vgg16_bn(pretrained=True).cuda()
    for test_mode in [False, True]:
        if test_mode:
            ignore_count = 10
            i = 0
            while len(test_clusters) < num_test_clusters:
                if i >= ignore_count:
                    test_clusters.append(np.argmax(cluster_scores))
                i += 1
                cluster_scores[np.argmax(cluster_scores)] = 0
            exemplars = {cluster: [] for cluster in test_clusters}
        num_images = num_test_images if test_mode else num_validation_images
        for image_path in tqdm(images[:num_images], file=sys.stdout):
            img = load_image(image_path, gpu=True)
            features = {layer: [] for layer in layers}
            handle = hook_layer(model, layer, features[layer])
            try:
                with torch.no_grad():
                    model.forward(img)
            except RuntimeError:
                continue
            deltas = kmeans.transform(features[layer][-1])
            cluster_assignments = np.argmin(deltas, axis=1)
            cluster_distances = np.min(deltas, axis=1)
            if test_mode:
                image_words[image_path][layer].extend(
                    list(cluster_assignments[cluster_distances < mean_sum - 1 * mean_std]))
                for cluster_num, cluster_id in enumerate(test_clusters):
                    idx = 0
                    cluster_assignments = list(cluster_assignments)
                    while cluster_id in cluster_assignments[idx:]:
                        idx = cluster_assignments[idx:].index(cluster_id) + idx
                        # assert cluster_assignments[idx] == cluster_id
                        if len(exemplars[cluster_id]) < num_exemplars \
                                or cluster_distances[idx] < exemplars[cluster_id][-1][1]:
                            try:
                                img = get_receptive_field(cluster_id, image_path, idx, features[layer][-1].shape[0])
                                if len(exemplars[cluster_id]) < num_exemplars:
                                    exemplars[cluster_id].append((img, cluster_distances[idx]))
                                else:
                                    exemplars[cluster_id][-1] = (img, cluster_distances[idx])
                            except TypeError as e:
                                print('\nfailed to display image:', e)
                        idx += 1
                    exemplars[cluster_id].sort(
                        key=lambda x: x[1])  # We wait until now to sort so we have <=1 example/image
            else:
                mean_sum += np.mean(cluster_distances) / num_images
                mean_std += np.std(cluster_distances) / num_images
                n += 1
                if cluster_scores is None:
                    cluster_scores = np.bincount(cluster_assignments, minlength=deltas.shape[1])
                else:
                    new_scores = np.bincount(cluster_assignments, minlength=deltas.shape[1])
                    cluster_scores += new_scores
            handle.remove()
            del deltas, cluster_assignments, cluster_distances, features, img
            torch.cuda.empty_cache()
    if compute_mean_diff:
        print(mean_sum)
        print(mean_std)
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
    del kmeans, exemplars, model

with open('hierarchical-corpus.txt', 'w') as out_file:
    with open('hierarchical-corpus-with-hints.txt', 'w') as hinted_out_file:
        for image in image_words.keys():
            word_tokens = []
            hinted_tokens = []
            offset = 0
            for i, layer in enumerate(image_words[image].keys()):
                words = image_words[image][layer]  # type: List[int]
                new_tokens = [f'{word + offset}:{words.count(word)}' for word in set(words)]
                word_tokens.extend(new_tokens)
                new_tokens = [token + f':{len(image_words[image].keys()) - i - 1}' for token in new_tokens]
                hinted_tokens.extend(new_tokens)
                offset += vocab_sizes[i]
            if len(word_tokens) < 1:
                continue
            out_file.write(f'{len(word_tokens)} {" ".join(word_tokens)}\n')
            hinted_out_file.write(f'{len(hinted_tokens)} {" ".join(hinted_tokens)}\n')
