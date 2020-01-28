import numpy as np
import cvxpy as cp
class svm_model_cvxpy:
    def __init__(self, m,n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.n_class = n_class
        
        # multiplier
        self.a = [cp.Variable(shape=(m,1),pos=True) for i in range(self.n_svm)]
        # bias
        self.b = np.zeros((self.n_svm,1))

        
        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of poly kernel: lambda x,y:  torch.matmul(x,y.T)**2
        self.kernel = lambda x,y:  x @ y.T
        
        
        # Binary setting for every SVM, 
        # Mij says the SVMj should give 
        # Mij label to sample with class i
        self.lookup_matrix=np.zeros((self.n_class, self.n_svm))
        
        # The two classes SVMi concerns, 
        # lookup_class[i]=[pos, neg]
        self.lookup_class=np.zeros((self.n_svm, 2))
        
        k=0
        for i in range(n_class-1):
            for j in range(i+1,n_class):
                self.lookup_class[k, 0]=i
                self.lookup_class[k, 1]=j
                k += 1

        for i in range(n_class):
            for j in range(self.n_svm):
                if i == self.lookup_class[j,0] or i == self.lookup_class[j,1]:
                    if self.lookup_class[j, 0]==i:
                        self.lookup_matrix[i,j]=1.0
                    else:
                        self.lookup_matrix[i,j]=-1.0
    def fit(self, x, y_multiclass, kernel=lambda x,y:  (x @ y.T)**3, C=0.001):
        y_multiclass=y_multiclass.reshape((-1,1))
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        self.C = C
        # weight
        self.w = np.zeros((self.n_svm,x.shape[1]))
        for k in range(self.n_svm):
            print("training ",k,"th SVM in ",self.n_svm)
            y = self.cast(y_multiclass, k)
            yx = y*x
            G = kernel(yx, yx) # Gram matrix
            objective = cp.Maximize(cp.sum(self.a[k])-(1/2)*cp.quad_form(self.a[k], G))
            if not objective.is_dcp():
                print("Not solvable!")
                assert objective.is_dcp()
            
            constraints = [self.a[k] <= C, cp.sum(cp.multiply(self.a[k],y)) == 0] # box constraint
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            self.w[k,:] = self.get_w(k).reshape(-1)
            x_pos = x[y[:,0]==1,:]
            x_neg = x[y[:,0]==-1,:]
            b_min = np.min(self.kernel(self.w[k,:].reshape(1,-1),x_pos)) if x_pos.shape[0]!=0 else 0
            b_max = np.max(self.kernel(self.w[k,:].reshape(1,-1),x_neg)) if x_neg.shape[0]!=0 else 0
            self.b[k,0] = (-1/2)*(b_min + b_max)

    def predict(self,x):
        n_x = x.shape[0]
        k_predicts = np.zeros((self.n_svm, n_x))
        for k in range(self.n_svm):
            k_predicts[k,:] =  self.g_k(k, x).reshape(1,-1) #self.kernel(x, self.w[k]).reshape(1,-1) + self.b[k,0].reshape(1,1)
        result = np.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result.reshape(-1,1)
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).astype(np.float32) - (y==self.lookup_class[k, 1]).astype(np.float32)
        
    
    def g_k(self,k,xi):
        # The prediction of SVMk, xi[1,d]
        y = self.cast(self.y_multiclass, k)
        a = self.a[k].value.reshape(-1,1)
        wTx =  self.kernel(xi, self.x) @ (y * a)
        return wTx + self.b[k,0].reshape(1,1)
    
    
    def get_w(self, k):
        y = self.cast(self.y_multiclass, k)
        a = self.a[k].value.reshape(-1,1)
        return np.sum(a*y*self.x,0).reshape(-1,1)
    
    def get_svms(self):
        for k in range(self.n_svm):
            sk = 'g' + str(self.lookup_class[k, 0]) + str(self.lookup_class[k, 1]) + '(x)='
            w = self.get_w(k)
            for i in range(w.shape[0]):
                sk += "{:.3f}".format(w[i,0].item()) + ' x' + "{:d}".format(i) +' + '
            sk += "{:.3f}".format(self.b[k,0].item())
            print(sk)
            
    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors, 
        # test error shouldn't be greater than it if traing converge
        a_matrix = np.stack([i.value for i in self.a],0)
        return np.sum((0.0<a_matrix) & (a_matrix<self.C)).astype(np.float32)/(self.n_svm*self.m)
    

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_classification
data = load_iris()
colors = ['red','green','blue','yellow']
data_x,data_y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,n_clusters_per_class=1, n_classes=4,class_sep=2)
fig = plt.figure()
fig = plt.scatter(data_x[:,0],data_x[:,1],c=data_y, cmap=ListedColormap(colors), marker='o')
m = len(data_x)
c = len(np.unique(data_y))
svm = svm_model_cvxpy(m,c)
svm.fit(data_x,data_y,lambda x,y:  (x @ y.T)**3, 0.001)
from mlxtend.plotting import plot_decision_regions
x=np.linspace(-4,4,100)
test_x = np.array(np.meshgrid(x,x)).T.reshape(-1,2)
test_y = svm.predict(test_x).reshape(-1)
scatter_kwargs = {'alpha': 0.0}
fig =plot_decision_regions(test_x, test_y, clf=svm,scatter_kwargs=scatter_kwargs)
xx = np.linspace(-4,4,10)
for i in range(svm.n_svm):

    ak = svm.a[i].value.reshape(-1)
    mask = (svm.C*0.0001< ak) & (ak<svm.C*(1-0.0001))
    fig.scatter(data_x[mask, 0]+i/8, data_x[mask,1],marker=4)
plt.show()

print(np.sum(svm.predict(data_x).reshape(-1)==data_y)/len(data_y))
