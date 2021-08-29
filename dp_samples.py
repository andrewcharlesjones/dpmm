import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet

import matplotlib.animation as animation
import matplotlib.image as mpimg
from os.path import join as pjoin
import os

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def dirichlet_pdf(x, alpha):

    try:
        return dirichlet.pdf(x, alpha)
    except:
        return None


def dirichlet_pdf_mv(X, alpha):
    return np.array([dirichlet_pdf(X[ii, :], alpha) for ii in range(X.shape[0])])


xlimits = [0, 1]
ylimits = [0, 1]
numticks = 100
x1s = np.linspace(*xlimits, num=numticks)
x2s = np.linspace(*ylimits, num=numticks)
X1, X2 = np.meshgrid(x1s, x2s)
X = np.vstack([X1.ravel(), X2.ravel()]).T

# k = 3

# alpha = [10, 5, 5]
# X_probs = dirichlet_pdf_mv(X, alpha)


a1_list = np.linspace(1.0, 50.0, 10)
# import ipdb; ipdb.set_trace()
a1_list = np.append(a1_list, np.flip(a1_list))

for ii, a1 in enumerate(a1_list):
    a1 = round(a1, 1)
    alpha = [a1, 5, 5]
    # plt.figure(figsize=(5, 5))
    X_probs = dirichlet_pdf_mv(X, alpha)
    plt.scatter(X[:, 0], X[:, 1], c=X_probs)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"$\alpha = " + str(alpha) + r"$")
    plt.tight_layout()
    plt.savefig("./tmp/tmp{}.png".format(ii))
    plt.close()
# plt.show()


fig = plt.figure()
ims = []
for ii in range(len(a1_list)):
    fname = "./tmp/tmp{}.png".format(ii)
    img = mpimg.imread(fname)
    im = plt.imshow(img)
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    ims.append([im])
    os.remove(fname)

SAVE_DIR = (
    "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"
)
writervideo = animation.FFMpegWriter(fps=5)
ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save(
    pjoin(SAVE_DIR, "dirichlet_distribution_animation.mp4"),
    writer=writervideo,
    dpi=1000,
)
import ipdb

ipdb.set_trace()
