import torch
class svm_model_torch:
    def __init__(self, m, C, n_class):
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # of samples
        self.C = C # box constraint
        
        # multiplier
        self.a = torch.rand((self.n_svm,self.m), 
                            #device=torch.device("cuda"), 
                            requires_grad=True) * self.C
        # kernel function
        self.kernel = lambda x,y: torch.sum(x*y)
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
                        
    def fit(self, x_np, y_multiclass_np, iterations=10, kernel=lambda x,y: torch.sum(x*y)):
        x = torch.from_numpy(x_np)
        y_multiclass = torch.from_numpy(y_multiclass_np)
        self.x = x
        self.y_multiclass = y_multiclass
        self.kernel = kernel
        for iteration in range(iterations):
            for k in range(self.n_svm):
                y = self.cast(y_multiclass, k)
                for i in range(self.m - 1):
                    y1 = y[i,0]
                    y2 = y[i+1,0]
                    x1 = x[i,:].view(1,-1)
                    x2 = x[i+1,:].view(1,-1)
                    a1_old = self.a[k,i]
                    a2_old = self.a[k,i+1]
                    if not (y1==0 or y2==0):
                        if y1 != y2:
                            H = min(self.C, (self.C + a2_old-a1_old).item())
                            L = max(0, (a2_old-a1_old).item())
                        else:
                            H = min(self.C, (a2_old + a1_old).item())
                            L = max(0, (a2_old + a1_old - self.C).item())
                        
                        E1 = self.error_k(k, x1, y1,x,y)
                        E2 = self.error_k(k, x2, y2,x,y)
                        dx = x1 - x2
                        kappa = self.kernel(dx,dx)
                        delta = y2 * (E1-E2)/kappa
                        a2_new = a2_old + delta
                        self.a[k,i+1] = torch.clamp(a2_new, L, H)
                        self.a[k, i] = a1_old - y1 * y2 * (a2_old - self.a[k,i+1])
                
    def predict(self,x_np):
        x = torch.from_numpy(x_np)
        n_x = x.shape[0]
        k_predicts = torch.zeros((self.n_svm, n_x))
        for k in range(self.n_svm):
            for i in range(n_x):
                y = self.cast(self.y_multiclass, k)
                k_predicts[k, i] = self.g_k(k, x[i,:].view(1,-1), self.x, y)
        result = torch.argmax(torch.matmul(self.lookup_matrix, k_predicts ),axis=0)
        return result.reshape(-1,1)
        
    def cast(self, y, k):
        # cast the label of dataset to the P/N/0 SVMk concerns
        return (y==self.lookup_class[k][0]).float() - (y==self.lookup_class[k][1]).float()
        
    def g_k(self, k, xi, x, y):
        # The prediction of SVMk
#         y = self.cast(y_multiclass, k)
        a = self.a[k,:].view(-1,1)
        gx = (y * a) * self.kernel(xi, x) # kernel should broadcast xi [1,d] to [m,d]
        return torch.sum(gx)
    
    def error_k(self, k, xi, yi, x, y):
        return self.g_k(k,xi,x,y)-yi
        
