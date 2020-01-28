import numpy as np
import cvxpy as cp
class svm_model_cvxpy:
    def __init__(self, m, C=0.1, n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.C = C # box constraint
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
    def fit(self, x, y_multiclass, kernel=lambda x,y:  (x @ y.T)**3):
        y_multiclass=y_multiclass.reshape((-1,1))
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
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
            
            constraints = [self.a[k] <= self.C, cp.sum(cp.multiply(self.a[k],y)) == 0] # box constraint
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
