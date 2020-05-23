import torch
def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        X12norm = torch.sum(x1**2,1,keepdims=True)-2*x1@x2.T+torch.sum(x2**2,1,keepdims=True).T
        return torch.exp(-X12norm/(2*sigma**2))
    return lambda x1,x2: rbf_kernel(x1,x2,sigma)

def poly(n=3):
    return lambda x1,x2: (x1 @ x2.T)**n

def grpf(sigma, d):
    return lambda x1,x2: ((d + 2*rbf(sigma)(x1,x2))/(2 + d))**(d+1)


