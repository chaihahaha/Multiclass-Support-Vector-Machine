import torch
import numpy as np
class svm_model_torch:
    def __init__(self, m, C, n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.C = C # box constraint
        
        # multiplier
        self.a = torch.rand((self.n_svm,self.m)) * self.C
        # bias
        self.b = torch.rand((self.n_svm,1))
        
        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of poly kernel: lambda x,y:  torch.matmul(x,y.T)**2
        self.kernel = lambda x,y:  torch.matmul(x,y.T)
        self.n_class = n_class
        
        # Binary setting for every SVM, 
        # Mij says the SVMj should give 
        # Mij label to sample with class i
        self.lookup_matrix=torch.zeros((self.n_class, self.n_svm))
        
        # The two classes SVMi concerns, 
        # lookup_class[i]=[pos, neg]
        self.lookup_class=dict()
        
        k=0
        for i in range(n_class-1):
            for j in range(i+1,n_class):
                self.lookup_class[k]=[i,j]
                k += 1

        for i in range(n_class):
            for j in range(self.n_svm):
                if i in self.lookup_class[j]:
                    if self.lookup_class[j][0]==i:
                        self.lookup_matrix[i,j]=1.0
                    else:
                        self.lookup_matrix[i,j]=-1.0
                        
    def fit(self, x_np, y_multiclass_np, iterations=10, kernel=lambda x,y: torch.matmul(x,y.T)):
        # use SMO algorithm to fit
        x = torch.from_numpy(x_np).float()
        y_multiclass = torch.from_numpy(y_multiclass_np)
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        for iteration in range(iterations):
            for k in range(self.n_svm):
                y = self.cast(y_multiclass, k)
                index, _ = torch.where(y!=0)
                for i in range(len(index)-1):
                    y1 = y[index[i],0]
                    y2 = y[index[i+1],0]
                    x1 = x[index[i],:].view(1,-1)
                    x2 = x[index[i+1],:].view(1,-1)
                    a1_old = self.a[k,index[i]]
                    a2_old = self.a[k,index[i+1]]

                    if y1 != y2:
                        H = min(self.C, (self.C + a2_old-a1_old).item())
                        L = max(0, (a2_old-a1_old).item())
                    else:
                        H = min(self.C, (a2_old + a1_old).item())
                        L = max(0, (a2_old + a1_old - self.C).item())
                    
                    E1 = self.error_k(k, x1, y1)
                    E2 = self.error_k(k, x2, y2)
                    dx = x1 - x2
                    kappa = self.kernel(dx,dx)
                    delta = y2 * (E1-E2)/kappa
                    
                    a2_new = a2_old + delta
                    self.a[k,index[i+1]] = torch.clamp(a2_new, L, H)
                    self.a[k, index[i]] = a1_old - y1 * y2 * (a2_old - self.a[k,index[i+1]])

                    a1_new = self.a[k,index[i]]
                    a2_new = self.a[k,index[i+1]]
                    
                    wTx1 = self.g_k_nobias(k, self.x)[y==-1]
                    wTx2 = self.g_k_nobias(k, self.x)[y==1]
                    self.b[k,0] = (torch.max(wTx1) if len(wTx1)!=0 else 0
                              + torch.min(wTx2) if len(wTx2)!=0 else 0)/(-2.0)          
                
    def predict(self,x_np):
        x = torch.from_numpy(x_np).float()
        n_x = x.shape[0]
        k_predicts = torch.zeros((self.n_svm, n_x))
        for k in range(self.n_svm):
            for i in range(n_x):
                k_predicts[k, i] = self.g_k(k, x[i,:].view(1,-1))
        result = torch.argmax(torch.matmul(self.lookup_matrix, k_predicts ),axis=0)
        return result.reshape(-1,1)
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k][0]).float() - (y==self.lookup_class[k][1]).float()
        
    def g_k_nobias(self, k, xi):
        # The prediction of SVMk, xi[1,d]
        y = self.cast(self.y_multiclass, k)
        a = self.a[k,:].view(-1,1)
        gx =  torch.matmul(self.kernel(xi, self.x), (y * a))
        return gx
    
    def g_k(self,k,xi):
        return self.g_k_nobias(k,xi) + self.b[k,0].view(1,1)
    
    def error_k(self, k, xi, yi):
        return self.g_k(k,xi)-yi.view(1,1)
    
    def get_w(self, k):
        y = self.cast(self.y_multiclass, k)
        a = self.a[k,:].view(-1,1)
        return torch.sum(a*y*self.x,0).view(-1,1)
    
    def get_svms(self):
        for k in range(self.n_svm):
            sk = 'g' + str(self.lookup_class[k][0]) + str(self.lookup_class[k][1]) + '(x)='
            w = self.get_w(k)
            for i in range(w.shape[0]):
                sk += "{:.3f}".format(w[i,0].item()) + ' x' + "{:d}".format(i) +' + '
            sk += "{:.3f}".format(self.b[k,0].item())
            print(sk)
        
