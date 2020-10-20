from utils import *
from svm_torch import *
train_x, train_y = preprocessing(*(load("training_images.npy", "training_labels.npy")))
n_train = 1000
svm = svm_model_torch(n_train,10)
svm.fit(train_x[:n_train],train_y[:n_train],1,kernel=grpf(10,3))
val_x, val_y = preprocessing(*(load("validation_images.npy", "validation_labels.npy")))
p_y = svm.predict(val_x)
print(sum(p_y==val_y)/len(val_y))
