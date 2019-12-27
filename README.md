# Multiclass Support Vector Machine Tensorflow

This is a multiclass SVM implmented on Tensorflow by Shitong CHAI. Copyright Reserved.

### Use increase_dims() to increase the dimension of input features

### Example

from svm import *

train_x=increase_dims(train_x)

svm = svm_model(number_of_classes,train_x.shape[1],learning_rate,regularization)

svm.fit(train_x,train_y,iterations)

svm.predict(test_x)
