# Multiclass Support Vector Machine Tensorflow

This is a multiclass SVM implmented on Tensorflow, Pytorch, cvxpy by Shitong CHAI. Copyright Reserved.

### FYI

The cvxpy version with sparse matrix is super efficient for memory.

### Use increase_dims() to increase the dimension of input features

### Example of Tensorflow version
```python
from svm_tf import *
import numpy as np
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel="rbf", gamma=1)
train_x = kpca.fit_transform(raw_X) # or train_x=increase_dims(raw_x)

svm = svm_model(number_of_classes,train_x.shape[1],learning_rate,regularization)

svm.fit(train_x,train_y,iterations)

svm.predict(test_x)
```
To save weights and load weights:
```python
svm.save("svm.pickle")
```

```python
svm=svm_model(n_classes,dimension)
svm.load("svm.pickle")

```

### Example of pytorch version
```python
import numpy as np
from svm_torch import *
data_x = np.array([[-1,0],[1,0],[0,1]])
data_y = np.array([[0],[1],[2]])
m = len(data_x)
c = len(np.unique(data_y))
svm = svm_model_torch(m,1,c)
svm.fit(data_x,data_y,1)

print(svm.predict(data_x))
svm.get_svms()
print(svm.a)
print(svm.get_avg_pct_spt_vec()) # the theoretical upper bound of generalization error
```

### Example of Sparse matrix cvxpy SVM
```python
import numpy as np
from svm_cvxpy_sparse import *
data_x = np.array([[-1,0],[1,0],[0,1]])
data_y = np.array([[0],[1],[2]])
m = len(data_x)
c = len(np.unique(data_y))
svm = svm_model_cvxpy(m,c)
svm.fit(data_x,data_y, kernel=poly(3), C=1e-3)

print(svm.predict(data_x))
print(svm.a)
print(svm.get_avg_pct_spt_vec()) # the theoretical upper bound of generalization error
```

