import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from gaussian_marginal_likelihood import compute_ml
from os.path import join as pjoin
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"

n = 30
n_iters = 300
burnin = 0
alpha = 1
R = 1

SAVE_GIF = False

def plot_gaussian(mean):
    xs = np.linspace(-5, 5, 100)
    ys = norm.pdf(xs, mean, 1)
    plt.plot(xs, ys, color="red")
    

## Generate data
X1 = np.random.normal(loc=-2, scale=0.5, size=n // 2)
X2 = np.random.normal(loc=2, scale=0.5, size=n // 2)
X = np.concatenate([X1, X2])

plt.figure(figsize=(5, 5))
plt.hist(X, 30)
plt.title("Data")
plt.xlabel("x")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "dpmm_data.png"))
plt.close()

## Initialize parameters
# cluster_means = [np.random.normal()] * n
cluster_means = np.random.normal(size=n)
param_history = np.empty((n_iters, n))
n_clusters_history = np.empty(n_iters)

param_history[0, :] = cluster_means
n_clusters_history[0] = len(np.unique(cluster_means))

param_count_history = {}

for ii in range(1, n_iters):

    means_unique, mean_counts = np.unique(cluster_means, return_counts=True)

    if ii > burnin:
        for kk, mm in enumerate(means_unique):
            if mm not in param_count_history:
                param_count_history[mm] = {}
            param_count_history[mm][ii] = mean_counts[kk]

    if SAVE_GIF:
        plt.hist(X, 30, color="blue", alpha=0.5)
        plt.title("Iter: {}".format(ii))
        for curr_mean in means_unique:
            plot_gaussian(curr_mean)
        plt.savefig("./tmp/tmp{}.png".format(ii))
        plt.close()

    for jj in range(n):

        

        ## Sample mean
        means_unique, mean_counts = np.unique(cluster_means, return_counts=True)

        # Remove this sample from the counts
        mean_counts -= (means_unique == cluster_means[jj]).astype(int)

        # Could draw an existing param or a new one
        possible_samples = np.arange(len(means_unique) + 1)

        

        # existingparam_prob = norm.logpdf(X[jj], loc=newsample, scale=1)

        ## Compute likelihood of this sample under each existing parameter
        sample_likelihoods = norm.pdf(X[jj], loc=means_unique, scale=1)

        ## Don't compute it for this sample's current parameter
        # sample_likelihoods[jj] = 0


        r_jj = compute_ml(X[jj])
        newsample_prob = alpha * r_jj
        
        sample_probs = np.append(sample_likelihoods * mean_counts, newsample_prob)

        ## Normalize
        sample_probs /= np.sum(sample_probs)

        ## Draw sample index
        # import ipdb; ipdb.set_trace()
        sample_idx = np.random.choice(possible_samples, p=sample_probs)
        
        if sample_idx == len(possible_samples) - 1:
            # Draw new sample from posterior of mean given this sample only
            ### mu ~ N(xj/2, 1/2)
            newsample = np.random.normal(X[jj] / 2.0, 0.5)
        else:
            newsample = means_unique[sample_idx]

        # import ipdb; ipdb.set_trace()


        cluster_means[jj] = newsample
        param_history[ii, jj] = cluster_means[jj]

    n_clusters_history[ii] = len(np.unique(cluster_means))

    # if SAVE_GIF:
    #     plt.savefig("./tmp/tmp{}.png".format(ii))
    #     plt.close()
    # import ipdb; ipdb.set_trace()


plt.figure(figsize=(7, 7))
for uniqueval in param_count_history.keys():
    curr_iters = param_count_history[uniqueval].keys()

    uniquevals = [uniqueval] * len(curr_iters)
    # plt.plot(curr_iters, uniquevals, color="blue", alpha=0.5)
    bucket_counts = np.array(list(param_count_history[uniqueval].values()))
    fillwidth = bucket_counts * 0.01
    plt.fill_between(
        curr_iters,
        uniquevals - fillwidth,
        uniquevals + fillwidth,
        alpha=0.5,
        color="blue",
    )
plt.xlabel("Iteration")
plt.ylabel(r"$\mu$ samples")
plt.title("DPMM samples")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "dpmm_samples_over_iters.png"))
# plt.show()
plt.close()
# import ipdb; ipdb.set_trace()


plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.hist(X, 30)
# plt.title("Data")
# plt.xlabel("x")
# plt.ylabel("Density")
plt.subplot(121)
plt.hist(np.ndarray.flatten(param_history[100:, :]), 30)
plt.title("DPMM samples")
plt.xlabel(r"$\mu$")
plt.ylabel("Density")
plt.subplot(122)
plt.title("Posterior, num. of components")
plt.xlabel("Number of components")
plt.ylabel("Density")
plt.hist(n_clusters_history[100:])
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "dpmm_sampled_posterior.png"))
# plt.show()
plt.close()


if SAVE_GIF:
    fig = plt.figure()
    ims = []
    for ii in range(1, n_iters):
        fname = "./tmp/tmp{}.png".format(ii)
        img = mpimg.imread(fname)
        im = plt.imshow(img)
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_xticks([])
        ims.append([im])
        os.remove(fname)

    writervideo = animation.FFMpegWriter(fps=5)
    ani = animation.ArtistAnimation(fig, ims, interval=200)
    ani.save(
        pjoin(SAVE_DIR, "dpmm_sampler_animation.mp4"),
        writer=writervideo,
        dpi=1000,
    )
import ipdb

ipdb.set_trace()
