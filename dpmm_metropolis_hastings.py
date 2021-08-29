import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

n = 40
n_iters = 300
burnin = 200
alpha = 1
R = 1

## Generate data
X1 = np.random.normal(loc=-2, scale=0.1, size=n // 2)
X2 = np.random.normal(loc=2, scale=0.1, size=n // 2)
X = np.concatenate([X1, X2])

## Initialize parameters
# cluster_means = [np.random.normal()] * n
cluster_means = np.random.normal(size=n)
param_history = np.empty((n_iters, n))
n_clusters_history = np.empty(n_iters)

param_count_history = {}

for ii in range(n_iters):

    means_unique, mean_counts = np.unique(cluster_means, return_counts=True)

    if ii > burnin:
        for kk, mm in enumerate(means_unique):
            if mm not in param_count_history:
                param_count_history[mm] = {}
            param_count_history[mm][ii] = mean_counts[kk]

    for jj in range(n):

        ## Sample mean
        means_unique, mean_counts = np.unique(cluster_means, return_counts=True)

        # Remove this sample from the counts
        mean_counts -= (means_unique == cluster_means[jj]).astype(int)

        # Could draw an existing param or a new one
        possible_samples = np.arange(len(means_unique) + 1)

        existingparam_prob = 1 / (n - 1 + alpha)
        newparam_prob = alpha / (n - 1 + alpha)
        sample_probs = np.append(existingparam_prob * mean_counts, newparam_prob)
        sample_probs /= np.sum(sample_probs)
        sample_idx = np.random.choice(possible_samples, p=sample_probs)

        if sample_idx == len(possible_samples) - 1:
            # Draw new sample from prior
            newsample = np.random.normal()
        else:
            newsample = means_unique[sample_idx]

        ## Compute acceptance probability
        newsample_likelihood = norm.logpdf(X[jj], loc=newsample, scale=1)
        newsample_prior = norm.logpdf(newsample, loc=0, scale=1)
        newsample_prob = newsample_likelihood + 1 / n * newsample_prior
        oldsample_likelihood = norm.logpdf(X[jj], loc=cluster_means[jj], scale=1)
        oldsample_prior = norm.logpdf(cluster_means[jj], loc=0, scale=1)
        oldsample_prob = oldsample_likelihood + 1 / n * oldsample_prior

        accept_margin = np.exp(newsample_prob - oldsample_prob)
        accept_prob = min(1, accept_margin)

        ## Save sample
        if np.random.uniform() < accept_prob:
            cluster_means[jj] = newsample

        param_history[ii, jj] = cluster_means[jj]

        # if ii > burnin and jj == 0:
        # 	print(cluster_means[jj])
    n_clusters_history[ii] = len(np.unique(cluster_means))


for uniqueval in param_count_history.keys():
    curr_iters = param_count_history[uniqueval].keys()

    uniquevals = [uniqueval] * len(curr_iters)
    # plt.plot(curr_iters, uniquevals, color="blue", alpha=0.5)
    bucket_counts = np.array(list(param_count_history[uniqueval].values()))
    # import ipdb; ipdb.set_trace()
    fillwidth = bucket_counts * 0.01
    plt.fill_between(
        curr_iters,
        uniquevals - fillwidth,
        uniquevals + fillwidth,
        alpha=0.5,
        color="blue",
    )
plt.show()
# import ipdb; ipdb.set_trace()


plt.figure(figsize=(21, 5))
plt.subplot(131)
plt.hist(X, 30)
plt.subplot(132)
plt.hist(np.ndarray.flatten(param_history[burnin:, :]), 30)
plt.subplot(133)
plt.hist(n_clusters_history[burnin:])
plt.show()
import ipdb

ipdb.set_trace()
