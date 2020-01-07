%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
colors = ['red','green','blue','purple']
data_x = np.array([[-2,1],[-2,2],[-1,1],[-1,2],[1,1],[1,2],[2,1],[2,2],[1,-1],[1,-2],[2,-1],[2,-2]])
data_y = np.array([[0],[0],[0],[0],[1],[1],[1],[1],[2],[2],[2],[2]]).reshape(-1)
fig = plt.figure()
fig = plt.scatter(data_x[:,0],data_x[:,1],c=data_y, cmap=ListedColormap(colors), marker='o')
m = len(data_x)
c = len(np.unique(data_y))
svm = svm_model_torch(m,1,c)
svm.fit(data_x,data_y,100)
from mlxtend.plotting import plot_decision_regions
x=np.linspace(-4,4,1000)
test_x = np.array(np.meshgrid(x,x)).T.reshape(-1,2)
test_y = svm.predict(test_x).reshape(-1)
scatter_kwargs = {'alpha': 0.0}
fig =plot_decision_regions(test_x, test_y, clf=svm,scatter_kwargs=scatter_kwargs)
xx = np.linspace(-4,4,10)
for i in range(svm.n_svm):
    w = svm.get_w(i)
    if w[1,0]==0:
        plt.axvline(x=(-svm.b[i,0]/w[0,0]).item())
    else:
        k = (- w[0,0]/w[1,0]).item()
        b = (- svm.b[i,0]/w[1,0]).item()
        fig.plot(xx,k*xx+b)
        fig.axis([-4,4,-4,4])
    ak = svm.a[i, :].numpy()
    mask = (0< ak) & (ak<svm.C.item())
    fig.scatter(data_x[mask, 0]+i/8, data_x[mask,1],marker=4)
plt.show()
