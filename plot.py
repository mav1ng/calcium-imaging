import matplotlib.pyplot as plt
import torch


def plot3Dembeddings(embeddings):
    xlist = []
    ylist = []
    zlist = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(embeddings.size(-1)):
        for k in range(embeddings.size(-2)):
            axis = embeddings.detach().numpy()[0, -1, :, i, k]
            xlist.append(axis[0])
            ylist.append(axis[1])
            zlist.append(axis[2])
    ax.scatter(xs=xlist, ys=ylist, zs=zlist, zdir='z', s=20)
    plt.show()
