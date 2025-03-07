# Mutual info scaling law
# Copyright (C) 2025 Zhuo Chen

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pickle
from math import log
import numpy as np
from scipy.special import logsumexp, digamma
from scipy.stats import linregress
import matplotlib.pyplot as plt

def G_func(n):
    return digamma(n) + (-1) ** n / 2 * (digamma((n + 1) / 2) - digamma(n / 2))

load_dir = 'meta-llama_Llama-3.1-405B-FP8/pg19_5000'

with open(f"{load_dir}/count_1gram.pkl", "rb") as f:
    counts = pickle.load(f)

total_counts = sum(counts.values())

entropy_single = log(total_counts) - sum(count * G_func(count) for count in counts.values()) / total_counts
entropy_single_err = np.sqrt(sum(count / total_counts * (G_func(count) - log(total_counts)) ** 2 for count in counts.values()) - entropy_single ** 2) / np.sqrt(total_counts)

del counts

np.savetxt(f'long_range_counts/{load_dir[:-5]}/entropies_single.txt', np.array([total_counts, entropy_single, entropy_single_err]))

print(total_counts, entropy_single)


ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
entropies = []
entropies_err = []

for d in ds:

    with open(f"long_range_counts/{load_dir[:-5]}/count_{d}distance.pkl", "rb") as f:
        counts_d = pickle.load(f)
    total_counts_d = sum(counts_d.values())
    entropy_d = log(total_counts_d) - sum(count * G_func(count) for count in counts_d.values()) / total_counts_d
    entropy_d_err = np.sqrt(sum(count / total_counts_d * (G_func(count) - log(total_counts_d)) ** 2 for count in counts_d.values()) - entropy_d ** 2) / np.sqrt(total_counts_d)

    entropy_d_naive = log(total_counts_d) - sum(count * log(count) for count in counts_d.values()) / total_counts_d
    entropy_d_digamma = log(total_counts_d) - sum(count * digamma(count) for count in counts_d.values()) / total_counts_d

    mi = entropy_single * 2 - entropy_d

    print(f"{d=}, {entropy_d=}, {mi=} {entropy_d_naive=} {entropy_d_digamma=}")
    print(total_counts_d)

    entropies.append(entropy_d)
    entropies_err.append(entropy_d_err)

np.savetxt(f'long_range_counts/{load_dir[:-5]}/entropies_d.txt', np.array([ds, entropies, entropies_err]).T)




# Use this to fit the bias corrected version
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

class LogLogBiasRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        def func(params):
            log_A, beta, c = params
            c = sigmoid(c) * y.min()
            return ((np.log(y - c) - log_A - beta * np.log(x))**2).mean()
        res = minimize(func, x0=(0, -0.4, 0))
        self.log_A, self.beta, self.c = res.x
        self.A = np.exp(self.log_A)
        self.c = sigmoid(self.c) * y.min()
    
    def predict(self, x):
        return self.A * x**self.beta + self.c
