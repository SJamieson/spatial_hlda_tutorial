import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from hlda.sampler import NCRPNode


class Node:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __getitem__(self, item):
        if item == 0:
            return self.name
        elif item == 1:
            return self.children


def build_graph_from_tree(G: nx.Graph, level, image_pattern):
    for node in level:
        G.add_node(node[0], image=image_pattern.format(node[0]))
        if node[1] is not None:
            build_graph_from_tree(G, node[1], image_pattern)
            for child in node[1]:
                G.add_edge(node[0], child[0])


def plot_image_hierarchy(fig: plt.Figure, G: nx.Graph):
    # Adapted from https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes
    ax = fig.gca()
    ax.set_aspect('equal', adjustable='box')

    write_dot(G, 'test.dot')
    pos = graphviz_layout(G, prog='dot', args='-Nsep="+1000,+1000";')
    nx.draw_networkx_edges(G, pos, ax=ax)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    piesize = 0.2  # this is the image size
    p2 = piesize / 2.0
    for n in G:
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - p2, ya - p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(plt.imread(G.node[n]['image']))
        a.set_xticks([])
        a.set_yticks([])
        # a.set_title(n.name)

    ax.axis('off')


def word_counts_to_image(word_counts, save=None):
    image = word_counts.reshape((5, 5))
    image.reshape((5, 5))
    if np.max(image) == 0:
        image = (255 - image).astype(np.uint8)
    else:
        image = (255 - image * (255 / np.max(image))).astype(np.uint8)
    if save is not None:
        plt.imsave(save + '.png', image, cmap='gray', vmin=0, vmax=255)
    return image


def build_tree_from_hlda(node: NCRPNode, topics=None):
    if topics is None:
        topics = {}
    topics[node.node_id] = word_counts_to_image(node.word_counts, save=f'output/inference-z{node.node_id}')
    if node.children is not None:
        children = []
        for child in node.children:
            child_node, topics = build_tree_from_hlda(child, topics=topics)
            children.append(child_node)
    else:
        children = None
    return Node(node.node_id, children), topics


def build_tree_from_file(filename: str, output_dir: str):
    topics = {}
    nodes = {}
    root = None
    with open(filename, 'r') as tree_file:
        for line in tree_file:
            tokens = line.split(' ')
            topic_id = tokens[0]
            if root is None:
                root = topic_id
            if topic_id not in nodes:
                nodes[topic_id] = []
            words = np.ndarray((25,))
            for token in tokens[1:-1]:
                if ':' not in token:
                    nodes[topic_id].append(token)
                else:
                    word, count = token.split(':')
                    words[int(word)] = int(count)
            topics[topic_id] = word_counts_to_image(words, save=output_dir + f'/inference-z{topic_id}')

    def get_node(node):
        children = [] if len(nodes[node]) > 0 else None
        for child in nodes[node]:
            children.append(get_node(child))
        return Node(node, children)

    return get_node(root), topics
