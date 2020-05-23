from svm_torch import svm_model_torch
from kernels import rbf, poly, grpf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_classification
colors = ['red','green','blue','yellow']
data_x,data_y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,n_clusters_per_class=1, n_classes=4,class_sep=2)
fig = plt.figure()
fig = plt.scatter(data_x[:,0],data_x[:,1],c=data_y, cmap=ListedColormap(colors), marker='o')
m = len(data_x)
c = len(np.unique(data_y))
svm = svm_model_torch(m,c)
svm.fit(data_x,data_y, 1, 10, grpf(1,3))
from mlxtend.plotting import plot_decision_regions
x=np.linspace(-4,4,100)
test_x = np.array(np.meshgrid(x,x)).T.reshape(-1,2)
test_y = svm.predict(test_x).reshape(-1)
scatter_kwargs = {'alpha': 0.0}
fig =plot_decision_regions(test_x, test_y, clf=svm,scatter_kwargs=scatter_kwargs)
xx = np.linspace(-4,4,10)
for i in range(svm.n_svm):

    ak = svm.a[i,:].reshape(-1)
    mask = (svm.C*0.0001< ak) & (ak<svm.C*(1-0.0001))
    fig.scatter(data_x[mask, 0]+i/8, data_x[mask,1],marker=4)
plt.show()
