import torch
import numpy as np

def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        X12norm = torch.sum(x1**2,1,keepdims=True)-2*x1@x2.T+torch.sum(x2**2,1,keepdims=True).T
        return torch.exp(-X12norm/(2*sigma**2))
    return lambda x1,x2: rbf_kernel(x1,x2,sigma)

def poly(n=3):
    return lambda x1,x2: (x1 @ x2.T)**n

def grpf(sigma, d):
    return lambda x1,x2: ((d + 2*rbf(sigma)(x1,x2))/(2 + d))**(d+1)

# Neural Tangent Kernel

import jax.numpy as jnp

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import neural_tangents as nt
from neural_tangents import stax

ResBlock = stax.serial(
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Erf(),
            stax.Dense(512, W_std=1.1, b_std=0),
        ),
        stax.Identity()
    ),
    stax.FanInSum()
)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=1, b_std=0),
    ResBlock, ResBlock, stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')
ntk_kernel = lambda x1,x2: np.array(kernel_fn(x1,x2,'ntk'))
