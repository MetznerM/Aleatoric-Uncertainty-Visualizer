#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:51:34 2023

@author: max
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

iris = np.load('iris/data/iris.npy')
X = pd.DataFrame(iris[:, :-1])
y = pd.DataFrame(iris[:, -1])

column_data = X[0]
print(min(column_data))
print(max(column_data))

sns.set_palette("plasma", n_colors=1) #funktionuert erst beim 2ten mal?
sns.kdeplot(data=column_data, fill=True)
plt.title(f'KDE für Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Dichte')

plt.show()