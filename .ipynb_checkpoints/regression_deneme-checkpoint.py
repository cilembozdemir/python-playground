# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:35:54 2021

@author: didar
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_excel (r'C:/Users/didar/Downloads/reg.xlsx')
print (df)

model = LinearRegression()



z=np.array(df[['Rainfall (mm)', 'Umbrellas sold']])
m=np.array(df[['Rainfall (mm)', 'Umbrellas sold']])

plt.scatter(z,m)
plt.show()



model.fit(z, m)

r_sq = model.score(z, m)
print('coefficient of determination:', r_sq)