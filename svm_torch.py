import torch
from random import shuffle
from kernels import rbf, poly, grpf
class svm_model_torch:
    def __init__(self, m, n_class, device="cpu"):
        self.device = device
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.n_class = n_class
        self.blacklist = [set() for i in range(self.n_svm)]

        # multiplier
        self.a = torch.zeros((self.n_svm,self.m), device=self.device) # SMO works only when a is initialized to 0
        # bias
        self.b = torch.zeros((self.n_svm,1), device=self.device)

        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of poly kernel: lambda x,y:  torch.matmul(x,y.T)**2
        self.kernel = lambda x,y:  torch.matmul(x,y.T)


        # Binary setting for every SVM,
        # Mij says the SVMj should give
        # Mij label to sample with class i
        self.lookup_matrix=torch.zeros((self.n_class, self.n_svm), device=self.device)

        # The two classes SVMi concerns,
        # lookup_class[i]=[pos, neg]
        self.lookup_class=torch.zeros((self.n_svm, 2), device=self.device)

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

    def fit(self, x_np, y_multiclass_np, C, iterations=1, kernel=rbf(1)):
        self.C = C # box constraint
        # use SMO algorithm to fit
        x = torch.from_numpy(x_np).float() if not torch.is_tensor(x_np) else x_np
        self.x = x.to(self.device)

        y_multiclass = torch.from_numpy(y_multiclass_np).view(-1,1) if not torch.is_tensor(y_multiclass_np) else y_multiclass_np
        y_multiclass=y_multiclass.view(-1)
        self.y_matrix = torch.stack([self.cast(y_multiclass, k) for k in range(self.n_svm)],0)
        self.kernel = kernel
        a = self.a
        b = self.b
        for iteration in range(iterations):
            print("Iteration: ",iteration)
            for k in range(self.n_svm):
                y = self.y_matrix[k, :].view(-1).tolist()
                index = [i for i in range(len(y)) if y[i]!=0]
                shuffle(index)
                traverse = []
                if index is not None:
                    traverse = [i for i in range(0, len(index)-1, 2)]
                    if len(index)>2:
                         traverse += [len(index)-2]
                for i in traverse:
                    if str(index[i])+str(index[i+1]) not in self.blacklist[k]:
                        y1 = y[index[i]]
                        y2 = y[index[i+1]]
                        x1 = x[index[i],:].view(1,-1)
                        x2 = x[index[i+1],:].view(1,-1)
                        a1_old = a[k,index[i]].clone()
                        a2_old = a[k,index[i+1]].clone()

                        if y1 != y2:
                            H = max(min(self.C, (self.C + a2_old-a1_old).item()),0)
                            L = min(max(0, (a2_old-a1_old).item()),self.C)
                        else:
                            H = max(min(self.C, (a2_old + a1_old).item()),0)
                            L = min(max(0, (a2_old + a1_old - self.C).item()),self.C)
                        E1 =  self.g_k(k, x1) - y1
                        E2 =  self.g_k(k, x2) - y2
                        a2_new = torch.clamp(a2_old + y2 * (E1-E2)/self.kernel(x1 - x2,x1 - x2), min=L, max=H)
                        a[k,index[i+1]] = a2_new

                        a1_new = a1_old - y1 * y2 * (a2_new - a2_old)
                        a[k, index[i]] = a1_new

                        b_old = b[k,0]
                        K11 = self.kernel(x1,x1)
                        K12 = self.kernel(x1,x2)
                        K22 = self.kernel(x2,x2)
                        b1_new = b_old - E1 + (a1_old-a1_new)*y1*K11+(a2_old-a2_new)*y2*K12
                        b2_new = b_old - E2 + (a1_old-a1_new)*y1*K12+(a2_old-a2_new)*y2*K22
                        if (0<a1_new) and (a1_new<self.C):
                            b[k,0] = b1_new
                        if (0<a2_new) and (a2_new<self.C):
                            b[k,0] = b2_new
                        if ((a1_new == 0) or (a1_new ==self.C)) and ((a2_new == 0) or (a2_new==self.C)) and (L!=H):
                            b[k,0] = (b1_new + b2_new)/2
                        if b_old == b[k,0] and a[k,index[i]] == a1_old and a[k,index[i+1]] == a2_old:
                            self.blacklist[k].add(str(index[i]) + str(index[i+1]))

    def predict(self,x_np):
        xp = torch.from_numpy(x_np) if not torch.is_tensor(x_np) else x_np
        xp = xp.float().to(self.device)
        k_predicts = (self.y_matrix.to(self.device) * self.a) @ self.kernel(xp,self.x).T  + self.b
        result = torch.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result.to("cpu").numpy()

    def cast(self, y, k):
        # cast the multiclass label of dataset to
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).float() - (y==self.lookup_class[k, 1]).float()


    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].reshape((-1,1))
        a = self.a[k,:].view(-1,1)
        wTx0 =  self.kernel(xi, self.x) @ (y * a)
        return wTx0


    def g_k(self,k,xi):
        # The prediction of SVMk, xi[1,d]
        return self.wTx(k,xi) + self.b[k,0].view(1,1)


    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors,
        # test error shouldn't be greater than it if traing converge
        return torch.sum((0.0<self.a) & (self.a<self.C)).float().item()/(self.n_svm*self.m)

