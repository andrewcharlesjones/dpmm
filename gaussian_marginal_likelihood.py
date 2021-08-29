import numpy as np
import matplotlib.pyplot as plt

prior_mean = 0
likelihood_variance = 1

a = (likelihood_variance + 1) ** 2
c1 = 1 / (2 * np.pi * np.sqrt(likelihood_variance))


def compute_ml(x):
    b = likelihood_variance * x
    c = likelihood_variance * x ** 2
    c2 = np.exp(0.5 * a * (b ** 2 / a ** 2 - c / a))
    gaussian_integral = np.sqrt(2 * np.pi) / np.sqrt(a)

    marginal_likelihood = c1 * c2 * gaussian_integral

    return marginal_likelihood


# X = np.linspace(-5, 5, 100)
# mls = np.array([compute_ml(x) for x in X])
# # print(marginal_likelihood)
# plt.plot(X, mls)
# plt.show()

# import ipdb

# ipdb.set_trace()
