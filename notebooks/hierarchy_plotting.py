import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

def plot_image_hierarchy(fig : plt.Figure, G : nx.Graph):
    # Adapted from https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes
    ax=fig.gca()
    ax.set_aspect('equal', adjustable='box')

    write_dot(G,'test.dot')
    pos =graphviz_layout(G, prog='dot', args='-Nsep="+1000,+1000";')
    nx.draw_networkx_edges(G,pos,ax=ax)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    piesize=0.2 # this is the image size
    p2=piesize/2.0
    for n in G:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(plt.imread(G.node[n]['image']))
        a.set_xticks([])
        a.set_yticks([])
        # a.set_title(n.name)

    ax.axis('off')