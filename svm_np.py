import numpy as np
from random import shuffle
from sklearn.utils import shuffle as shuffle_ds
def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        X12norm = np.sum(x1**2,1,keepdims=True)-2*x1@x2.T+np.sum(x2**2,1,keepdims=True).T
        return np.exp(-X12norm/(2*sigma**2))
    return lambda x1,x2: rbf_kernel(x1,x2,sigma)

def poly(n=3):
    return lambda x1,x2: (x1 @ x2.T)**n

def grpf(sigma, d):
    return lambda x1,x2: ((d + 2*rbf(sigma)(x1,x2))/(2 + d))**(d+1)

class svm_model_np:
    def __init__(self, m, n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.n_class = n_class
        self.blacklist = [set() for i in range(self.n_svm)]

        # multiplier
        self.a = np.zeros((self.n_svm,self.m)) # SMO works only when a is initialized to 0
        # bias
        self.b = np.zeros((self.n_svm,1))

        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of poly kernel: lambda x,y:  np.matmul(x,y.T)**2
        self.kernel = lambda x,y:  np.matmul(x,y.T)


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

    def fit(self, x, y_multiclass_np, C, iterations=1, kernel=rbf(1)):
        x, y_multiclass_np = shuffle_ds(x,y_multiclass_np)
        self.C = C # box constraint
        # use SMO algorithm to fit
        self.x = x

        y_multiclass=y_multiclass_np.reshape(-1)
        self.y_matrix = np.stack([self.cast(y_multiclass, k) for k in range(self.n_svm)],0)
        self.kernel = kernel
        a = self.a
        b = self.b
        for iteration in range(iterations):
            print("Iteration: ",iteration)
            for k in range(self.n_svm):
                y = self.y_matrix[k, :].reshape(-1).tolist()
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
                        x1 = x[index[i],:].reshape(1,-1)
                        x2 = x[index[i+1],:].reshape(1,-1)
                        a1_old = a[k,index[i]].copy()
                        a2_old = a[k,index[i+1]].copy()

                        if y1 != y2:
                            H = max(min(self.C, (self.C + a2_old-a1_old).item()),0)
                            L = min(max(0, (a2_old-a1_old).item()),self.C)
                        else:
                            H = max(min(self.C, (a2_old + a1_old).item()),0)
                            L = min(max(0, (a2_old + a1_old - self.C).item()),self.C)
                        E1 =  self.g_k(k, x1) - y1
                        E2 =  self.g_k(k, x2) - y2
                        a2_new = np.clip(a2_old + y2 * (E1-E2)/self.kernel(x1 - x2,x1 - x2), a_min=L, a_max=H)
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
                            self.blacklist[k].add(str(index[i+1]) + str(index[i]))

    def predict(self,xp):
        k_predicts = (self.y_matrix * self.a) @ self.kernel(xp,self.x).T  + self.b
        result = np.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result

    def cast(self, y, k):
        # cast the multiclass label of dataset to
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).astype(float) - (y==self.lookup_class[k, 1]).astype(float)


    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].reshape((-1,1))
        a = self.a[k,:].reshape(-1,1)
        wTx0 =  self.kernel(xi, self.x) @ (y * a)
        return wTx0


    def g_k(self,k,xi):
        # The prediction of SVMk, xi[1,d]
        return self.wTx(k,xi) + self.b[k,0].reshape(1,1)


    def get_w(self, k):
        y = self.cast(self.y_multiclass, k)
        a = self.a[k,:].reshape(-1,1)
        return np.sum(a*y*self.x,0).reshape(-1,1)

    def get_svms(self):
        for k in range(self.n_svm):
            sk = 'g' + str(self.lookup_class[k, 0].item()) + str(self.lookup_class[k, 1].item()) + '(x)='
            w = self.get_w(k)
            for i in range(w.shape[0]):
                sk += "{:.3f}".format(w[i,0].item()) + ' x' + "{:d}".format(i) +' + '
            sk += "{:.3f}".format(self.b[k,0].item())
            print(sk)

    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors,
        # test error shouldn't be greater than it if traing converge
        return np.sum((0.0<self.a) & (self.a<self.C))/(self.n_svm*self.m)
