import numpy as np
from scipy import sparse
import cvxpy as cp
def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        m=x1.shape[0]
        n=x2.shape[0]
        d=x1.shape[1]
        result = sparse.lil_matrix((m,n))
        for i in range(m):
            for j in range(n):
                substraction = x1[i,:] - x2[j,:]
                if sparse.issparse(substraction):
                    substraction = substraction.A
                result[i,j] = np.exp(-np.sum(substraction**2)/(2*sigma**2))
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
        y_multiclass=y_multiclass.reshape(-1).astype(np.float64)
        self.x = x.astype(np.double)
        self.m = len(x)
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        self.C = C
        ys = [sparse.csr_matrix(self.cast(y_multiclass, k)) for k in range(self.n_svm)]
        self.y_matrix = sparse.vstack(ys)
        del ys
        for k in range(self.n_svm):
            print("training ",k,"th SVM in ",self.n_svm)
            y = self.y_matrix[k, :].reshape((-1,1))#.tocsr()
            yx = y.multiply(x).tolil()
            G = kernel(yx, yx) # Gram matrix
            
            compensate = (sparse.eye(self.m)*1e-5).astype(np.float64)
            G = (G + compensate)
            objective = cp.Maximize(cp.sum(self.a[k])-(1/2)*cp.quad_form(self.a[k], G))
            
            if not objective.is_dcp():
                print("Not solvable!")
                assert objective.is_dcp()
            constraints = [self.a[k] <= C, cp.sum(cp.multiply(self.a[k],y)) == 0] # box constraint
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            x_pos = x[y.A[:,0]==1,:]
            x_neg = x[y.A[:,0]==-1,:]
            b_min = -np.min(self.wTx(k,x_pos)) if x_pos.shape[0]!=0 else 0
            b_max = -np.max(self.wTx(k,x_neg)) if x_neg.shape[0]!=0 else 0
            self.b[k,0] = (1/2)*(b_min + b_max)
        self.a_matrix = np.stack([i.value.reshape(-1) for i in self.a],0)
        self.a_matrix = sparse.csr_matrix(self.a_matrix)

    def predict(self,xp):
        k_predicts = (self.y_matrix.multiply(self.a_matrix)) @ self.kernel(xp,self.x).T  + self.b
        result = np.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).astype(np.float64) - (y==self.lookup_class[k, 1]).astype(np.float64)
        
    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].reshape((-1,1))
        a = self.a[k].value.reshape(-1,1)
        wTx0 =  self.kernel(xi, self.x) @ (y.multiply(a))
        return wTx0
     
    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors, 
        # test error shouldn't be greater than it if traing converge
        b1 = (0.0==self.a_matrix)
        b2 = self.a_matrix==self.C
        logical = b1+b2
        return 1-np.sum(logical).astype(np.float64)/(self.n_svm*self.m)
