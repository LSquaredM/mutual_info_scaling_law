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


from math import log
import numpy as np
from scipy.special import logsumexp
from scipy.stats import linregress
import matplotlib.pyplot as plt


load_dir = 'meta-llama_Llama-3.1-405B-FP8/pg19_5000'

res = {}
for batch_nb in range(10000):
    file_name = f'{load_dir}/batch_{batch_nb}.pkl'
    try:
        data = np.load(file_name, allow_pickle=True)
        for key in data.keys():
            if key not in res:
                res[key] = []
            if type(data[key]) == list:
                res[key] += data[key]
            elif type(data[key]) == int:
                res[key].append(data[key])
            else:
                assert False, f"Unknown type: {type(data[key])}"
    except FileNotFoundError:
        break


class LogLogRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = linregress(np.log(x), np.log(y))
        self.A = np.exp(self.intercept)
        self.beta = self.slope
    
    def predict(self, x):
        return np.exp(self.slope * np.log(x) + self.intercept)

for L_Lx_ratio in [2, 3, 4]:

    if L_Lx_ratio == 2:
        split_at_indices = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    else:
        split_at_indices = [2, 4, 8, 16, 32, 64, 128, 256]


    mi_clubs = []
    mi_club_errs = []

    for split_at in split_at_indices:
        term1 = np.array(res['no_split'])[:, split_at:L_Lx_ratio*split_at].sum(-1)
        term2 = np.array(res[f'split_{split_at}'])[:, split_at:L_Lx_ratio*split_at].sum(-1)
        mi_club = term1 - term2
        mi_club_err = mi_club.std(ddof=1) / np.sqrt(len(mi_club))
        mi_club = mi_club.mean()
        print(f"{split_at=}, {term1.mean()=}, {term2.mean()=}, {mi_club=} +- {mi_club_err}")
        mi_clubs.append(mi_club)
        mi_club_errs.append(mi_club_err)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    log_log_regression = LogLogRegression(np.array(split_at_indices), mi_clubs)
    ax1.errorbar(np.array(split_at_indices), mi_clubs, yerr=mi_club_errs, fmt='.')
    ax1.loglog(split_at_indices, log_log_regression.predict(np.array(split_at_indices)), label=f'{log_log_regression.A:.4f} * L^{log_log_regression.beta:.4f}')
    ax2.semilogx(np.array(split_at_indices), mi_clubs, '.')
    ax1.set_xlabel('L_x')
    ax1.set_ylabel('MI')
    ax1.legend()
    ax2.set_xlabel('L_x')
    ax2.set_ylabel('MI')
    fig.suptitle(f'MI club (L / L_x = {L_Lx_ratio})')
    fig.tight_layout()
    fig.savefig(f'{load_dir}/mi_club_Ly_Lx_{L_Lx_ratio}.png', dpi=300)

    np.savetxt(f'{load_dir}/mi_club_Ly_Lx_{L_Lx_ratio}.txt', np.array([split_at_indices, mi_clubs, mi_club_errs]).T)
