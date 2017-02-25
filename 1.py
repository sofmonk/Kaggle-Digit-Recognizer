import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
pred = rf.predict(test)
rf.score(train,target)

# np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

submission = pd.DataFrame({
        "ImageId": list(range(1,len(pred)+1)),
        "Label": pred
    })
submission.to_csv("submit.csv", index=False, header=True)