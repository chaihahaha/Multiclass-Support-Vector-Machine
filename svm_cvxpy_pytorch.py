import torch 
import cvxpy as cp
def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        m=len(x1)
        n=len(x2)
        d=x1.shape[1]
        x1 = x1.reshape((m,1,d))
        x2 = x2.view(1,n,d)
        result = torch.sum((x1-x2)**2,2)
        result = torch.exp(-result/(2*sigma**2))
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
        self.b = torch.zeros((self.n_svm,1),device="cuda")

        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of kernels: rbf(1.0), poly(3)
        self.kernel = rbf(1)
        
        # Binary setting for every SVM, 
        # Mij says the SVMj should give 
        # Mij label to sample with class i
        self.lookup_matrix=torch.zeros((self.n_class, self.n_svm),device="cuda")
        
        # The two classes SVMi concerns, 
        # lookup_class[i]=[pos, neg]
        self.lookup_class=torch.zeros((self.n_svm, 2))
        
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
    def fit(self, x_np, y_multiclass_np, kernel=rbf(1), C=0.001):
        x = torch.from_numpy(x_np) if not torch.is_tensor(x_np) else x_np
        x = x.float().to("cuda")
        y_multiclass = torch.from_numpy(y_multiclass_np) if not torch.is_tensor(y_multiclass_np) else y_multiclass_np
        y_multiclass = y_multiclass.to("cpu").view(-1)
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        self.C = C
        self.y_matrix = torch.stack([self.cast(y_multiclass, k) for k in range(self.n_svm)],0)
        for k in range(self.n_svm):
            print("training ",k,"th SVM in ",self.n_svm)
            y = self.y_matrix[k, :].view(-1,1)
            yx = y.to("cuda")*x
            G = kernel(yx, yx).to("cpu") # Gram matrix
            G = G + torch.eye(G.shape[0])*1e-5
            objective = cp.Maximize(cp.sum(self.a[k])-(1/2)*cp.quad_form(self.a[k], G))
#             if not objective.is_dcp():
#                 print("Not solvable!")
#                 assert objective.is_dcp()
            constraints = [self.a[k] <= C, cp.sum(cp.multiply(self.a[k],y)) == 0] # box constraint
            print(constraints)
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            x_pos = x[y[:,0]==1,:]
            x_neg = x[y[:,0]==-1,:]
            b_min = -torch.min(self.wTx(k,x_pos)) if x_pos.shape[0]!=0 else torch.tensor(0,device="cuda")
            b_max = -torch.max(self.wTx(k,x_neg)) if x_neg.shape[0]!=0 else torch.tensor(0,device="cuda")
            self.b[k,0] = (1/2)*(b_min + b_max)
        self.a_matrix = torch.stack([torch.from_numpy(i.value).float().view(-1) for i in self.a],0).to("cuda")

    def predict(self,x_np):
        xp = torch.from_numpy(x_np) if not torch.is_tensor(x_np) else x_np
        xp = xp.float().to("cuda")
        k_predicts = (self.y_matrix.to("cuda") * self.a_matrix) @ self.kernel(xp,self.x).T  + self.b
        result = torch.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result.to("cpu").numpy()
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).float() - (y==self.lookup_class[k, 1]).float()
        
    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].view(-1,1).to("cuda")
        a = torch.from_numpy(self.a[k].value).float().view(-1,1).to("cuda")
        wTx0 =  self.kernel(xi, self.x) @ (y * a)
        return wTx0
     
    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors, 
        # test error shouldn't be greater than it if traing converge
        return torch.sum((0.0<self.a_matrix) & (self.a_matrix<self.C)).float()/(self.n_svm*self.m)
