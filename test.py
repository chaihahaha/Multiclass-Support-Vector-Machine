from utils import *
from svm_torch import svm_model_torch
train_x, train_y = preprocessing(*(load("training_images.npy", "training_labels.npy")))
svm = svm_model_torch(5000,10)
svm.fit(train_x[:5000],train_y[:5000],1,kernel=grpf(10,3))
val_x, val_y = preprocessing(*(load("validation_images.npy", "validation_labels.npy")))
p_y = svm.predict(val_x)
print(sum(p_y==val_y)/len(val_y))
