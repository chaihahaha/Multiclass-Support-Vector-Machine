import numpy as np
def preprocessing(x, y):
    return x[:,:,0], np.argmax(y[:,:,0],axis=1)

def load(sx, sy):
    return np.load(sx), np.load(sy)
