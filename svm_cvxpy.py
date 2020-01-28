import numpy as np
import cvxpy as cp
def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        m=len(x1)
        n=len(x2)
        d=x1.shape[1]
        x1 = x1.reshape((m,1,d))
        x2 = x2.reshape((1,n,d))
        result = np.sum((x1-x2)**2,2)
        result = np.exp(-result/(2*sigma**2))
        return result
    return lambda x1,x2: rbf_kernel(x1,x2,sigma)

def poly(n=3):
    return lambda x1,x2: (x1 @ x2.T)**n

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
        # Example of kernels: rbf(1.0), poly(3)
        self.kernel = rbf(1)
        
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
    def fit(self, x, y_multiclass, kernel=rbf(1), C=0.001):
        y_multiclass=y_multiclass.reshape(-1)
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        self.C = C
        self.y_matrix = np.stack([self.cast(y_multiclass, k) for k in range(self.n_svm)],0)
        for k in range(self.n_svm):
            print("training ",k,"th SVM in ",self.n_svm)
            y = self.y_matrix[k, :].reshape((-1,1))
            yx = y*x
            G = kernel(yx, yx) # Gram matrix
            objective = cp.Maximize(cp.sum(self.a[k])-(1/2)*cp.quad_form(self.a[k], G))
            if not objective.is_dcp():
                print("Not solvable!")
                assert objective.is_dcp()
            constraints = [self.a[k] <= C, cp.sum(cp.multiply(self.a[k],y)) == 0] # box constraint
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            x_pos = x[y[:,0]==1,:]
            x_neg = x[y[:,0]==-1,:]
            b_min = -np.min(self.wTx(k,x_pos)) if x_pos.shape[0]!=0 else 0
            b_max = -np.max(self.wTx(k,x_neg)) if x_neg.shape[0]!=0 else 0
            self.b[k,0] = (1/2)*(b_min + b_max)
        self.a_matrix = np.stack([i.value.reshape(-1) for i in self.a],0)

    def predict(self,xp):
        k_predicts = (self.y_matrix * self.a_matrix) @ self.kernel(xp,self.x).T  + self.b
        result = np.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).astype(np.float32) - (y==self.lookup_class[k, 1]).astype(np.float32)
        
    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].reshape((-1,1))
        a = self.a[k].value.reshape(-1,1)
        wTx0 =  self.kernel(xi, self.x) @ (y * a)
        return wTx0
     
    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors, 
        # test error shouldn't be greater than it if traing converge
        return np.sum((0.0<self.a_matrix) & (self.a_matrix<self.C)).astype(np.float32)/(self.n_svm*self.m)
    
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
svm = svm_model_cvxpy(m,c)
svm.fit(data_x,data_y,rbf(1), 1e-3)
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
