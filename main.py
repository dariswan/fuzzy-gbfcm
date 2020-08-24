from GBFCM import FuzzyKohonen
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

#Parameter
alpha=0.2
alpha_decay= 1
max_epoch = 10
min_err = 0.01

#read data
data = pd.read_csv("tripadvisor_review10.csv")
data = data.drop("User ID", axis=1)

X = data.values

#shuffle data
train = X
np.random.shuffle(train)

#pengujian parameter
scores = []
models = []
klass = []

for kcount in range(2, 11):
    #Creating model
    print("jumlah kluster",kcount)
    gbfcm=FuzzyKohonen(size=(kcount,len(data.columns)), alpha=alpha, alpha_decay=alpha_decay)
    u, v = gbfcm(train, max_epoch, min_err)
    klas = gbfcm.forward(train)
    klass.append(klas)
    score = silhouette_score(train, klas)
    scores.append(score)
    models.append(gbfcm)

klass = pd.DataFrame(klass)
[print("jumlah klaster: %d, skor:%f" % (i+2, s)) for i, s in enumerate(scores)]

print(models[np.argmax(scores)].get_W())