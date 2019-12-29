import torch
import numpy as np
import time
class svm_model_torch:
    def __init__(self, m, C, n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.C = C # box constraint
        
        # multiplier
        self.a = torch.zeros((self.n_svm,self.m)) # SMO works only when a is initialized to 0
        # bias
        self.b = torch.zeros((self.n_svm,1))
        
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
                        
    def fit(self, x_np, y_multiclass_np, iterations=1, kernel=lambda x,y: torch.matmul(x,y.T)):
        # use SMO algorithm to fit
        x = torch.from_numpy(x_np).float() if not torch.is_tensor(x_np) else x_np
        y_multiclass = torch.from_numpy(y_multiclass_np) if not torch.is_tensor(y_multiclass_np) else y_multiclass_np
        x=x.to(self.device)
        y_multiclass=y_multiclass.to(self.device)
        
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        for iteration in range(iterations):
            for k in range(self.n_svm):
                y = self.cast(y_multiclass, k)
                index, _ = torch.where(y!=0)
                print("k=",k)
                s = time.time()
                traverse = [i for i in range(0, len(index)-1, 2)] + [len(index)-2]
                for i in traverse:
                    y1 = y[index[i],0].clone()
                    y2 = y[index[i+1],0].clone()
                    x1 = x[index[i],:].clone().view(1,-1)
                    x2 = x[index[i+1],:].clone().view(1,-1)
                    a1_old = self.a[k,index[i]].clone()
                    a2_old = self.a[k,index[i+1]].clone()
                    if y1 != y2:
                        H = max(min(self.C, (self.C + a2_old-a1_old).item()),0)
                        L = min(max(0, (a2_old-a1_old).item()),self.C)

                    else:
                        H = max(min(self.C, (a2_old + a1_old).item()),0)
                        L = min(max(0, (a2_old + a1_old - self.C).item()),self.C)

                    E1 = self.error_k(k, x1, y1)
                    E2 = self.error_k(k, x2, y2)

                    dx = x1 - x2
                    kappa = self.kernel(dx,dx)
                    delta = y2 * (E1-E2)/kappa
                    
                    a2_new_unclip = a2_old + delta
                    a2_new = torch.clamp(a2_new_unclip, min=L, max=H)
                    self.a[k,index[i+1]] = a2_new
                    a1_new = a1_old - y1 * y2 * (a2_new - a2_old)
                    self.a[k, index[i]] = a1_new
                    
                    b_old = self.b[k,0]
                    K11 = self.kernel(x1,x1)
                    K12 = self.kernel(x1,x2)
                    K22 = self.kernel(x2,x2)
                    b1_new = b_old - E1 + (a1_old-a1_new)*y1*K11+(a2_old-a2_new)*y2*K12
                    b2_new = b_old - E2 + (a1_old-a1_new)*y1*K12+(a2_old-a2_new)*y2*K22
                    if 0<a1_new<self.C:
                        self.b[k,0] = b1_new
                    if 0<a2_new<self.C:
                        self.b[k,0] = b2_new
                    if a1_new in [0,self.C] and a2_new in [0,self.C] and L!=H:
                        self.b[k,0] = (b1_new + b2_new)/2
                t = time.time()
                print(t-s)
                
    def predict(self,x_np):
        x = torch.from_numpy(x_np).float().to(self.device)
        n_x = x.shape[0]
        k_predicts = torch.zeros((self.n_svm, n_x),device=self.device)
        for k in range(self.n_svm):
            k_predicts[k,:] = self.g_k(k, x).view(1,-1)
        result = torch.argmax(torch.matmul(self.lookup_matrix, k_predicts ),axis=0)
        return result.reshape(-1,1).numpy()
    
    def to(self,device):
        self.device = device
        self.b = self.b.to(device)
        self.a = self.a.to(device)
        self.lookup_matrix = self.lookup_matrix.to(device)
        
    def cast(self, y, k):
        # cast the multiclass label of dataset to 
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k][0]).float() - (y==self.lookup_class[k][1]).float()
        
    def g_k_nobias(self, k, xi):
        # The prediction of SVMk, xi[1,d]
        y = self.cast(self.y_multiclass, k)
        a = self.a[k,:].view(-1,1)
        wTx =  torch.matmul(self.kernel(xi, self.x), (y * a))
        return wTx
    
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
        
