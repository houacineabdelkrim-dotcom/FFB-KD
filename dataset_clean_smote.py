from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from collections import Counter
df = pd.read_csv('./Datasets/dataset_cleaned.csv')
dataset = df.copy()
dataset.head()
x = dataset.iloc[:,:-3]
y = dataset.iloc[:,-3]
print("Before oversampling: ",Counter(y))
SMOTE = SMOTE()
# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(x, y)
print("After oversampling: ",Counter(y_train_SMOTE))
test1 = pd.concat([X_train_SMOTE, y_train_SMOTE], axis=1)
test1.to_csv("./Datasets/Label_Num_Smote.csv", index = False)