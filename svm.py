import tensorflow as tf
import numpy as np
import operator as op
from functools import reduce
import pickle

class svm_model:
    def __init__(self,n_class, dimension, learning_rate=1e-2, regularization=1):
        self.learning_rate=learning_rate
        self.n_class=n_class
        self.dimension=dimension
        self.w=[]
        self.b=[]
        self.model_y=[]
        self.loss=[]
        self.hinge_loss=[]
        self.regularization_loss=[]
        #prediction=[]
        self.gt_y=[]
        self.lookup_class=dict()
        self.w_model=np.random.normal(loc=0,scale=1,size=(self.ncr(n_class,2), dimension))
        self.b_model=np.random.normal(loc=0,scale=1,size=(self.ncr(n_class,2),1))
        self.lookup_matrix=np.zeros((n_class, self.ncr(n_class,2)),dtype=np.float32)
        self.batch_x=tf.placeholder(tf.float32,shape=(None,dimension),name="batch_x")
        self.batch_y=tf.placeholder(tf.float32,shape=(None,1),name="batch_y")
        self.w = tf.Variable(tf.random_uniform([self.ncr(n_class,2), self.dimension]))
        self.b = tf.Variable(tf.random_uniform([self.ncr(n_class,2),1]))
        k=0
        for i in range(n_class-1):
            for j in range(i+1,n_class):
                self.lookup_class[k]=[i,j]
                k += 1

        for i in range(n_class):
            for j in range(self.ncr(n_class,2)):
                if i in self.lookup_class[j]:
                    if self.lookup_class[j][0]==i:
                        self.lookup_matrix[i,j]=1.0
                    else:
                        self.lookup_matrix[i,j]=-1.0

        for i in range(self.ncr(n_class,2)):
            # idx is the index of all the samples svm i concerns
            
            # idx=tf.where[condition is true] 
            # tf.gather_nd(a,idx) 
            # is equivalent to a[condition is true]
            idx=tf.where(tf.keras.backend.any(tf.equal(self.batch_y,self.lookup_class[i]),1)) 
            
            # 1 x N matrix
            self.model_y.append(
                tf.tanh(
                    tf.matmul(tf.reshape(self.w[i,:],(1,dimension)),
                              tf.gather_nd(self.batch_x, idx), transpose_b=True)
                    + self.b[i,:])) 
            
            # 1 x N matrix
            self.gt_y.append(tf.reshape(self.zonp(tf.cast(
                tf.equal(tf.gather_nd(self.batch_y, idx),self.lookup_class[i][0]),
                tf.float32)),(1,-1)))
            
            self.hinge_loss.append(tf.reduce_mean(tf.maximum(0.0,1-tf.multiply(self.model_y[i],self.gt_y[i]))))
            self.regularization_loss.append(regularization * tf.norm(self.w[i,:], 2))
            self.loss.append(self.hinge_loss[i] + self.regularization_loss[i])
        self.total_loss = sum(self.loss)
        self.opt = tf.train.RMSPropOptimizer(learning_rate).minimize(self.total_loss)
        
    def ncr(self,n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom
    
    def zonp(self,zero_one):
        return 2*zero_one-1

    def fit(self,data_x,data_y,iter_time=1000, batch_size=1000):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([tf.assign(self.w,self.w_model),tf.assign(self.b,self.b_model)])
            
            total = data_y.shape[0]
            lower = 0
            upper = lower + batch_size
            
            counter = 0
            while counter < iter_time:
                lower = max(0,        (lower   +batch_size) % total)
                upper = min(total,      (lower +batch_size) % total)
                if lower<upper:
                    _,loss_val = sess.run([self.opt,self.total_loss], feed_dict={self.batch_x:data_x[lower:upper],self.batch_y:data_y[lower:upper]})
                    print("iteration:"+str(counter)+" loss:"+str(loss_val))
                    counter += 1
                
            w_model,b_model = sess.run([self.w,self.b])
        self.w_model = w_model
        self.b_model = b_model
        
        return w_model,b_model
    
    def predict(self,data_x):
        result = np.argmax(np.matmul(self.lookup_matrix, np.tanh(np.matmul(self.w_model,data_x.T)) ),axis=0)
        return result.reshape(-1,1)
    
    def save(self, file_name):
        with open(file_name,"wb") as f:
            pickle.dump((self.w_model,self.b_model), f)
    def load(self, file_name):
        with open(file_name,"rb") as f:
            w_tmp, b_tmp = pickle.load(f)
            if w_tmp.shape==(self.ncr(self.n_class,2), self.dimension) and b_tmp.shape == (self.ncr(self.n_class,2), 1):
                self.w_model, self.b_model = w_tmp, b_tmp
                print("Successfully loaded!")
            else:
                print("Loading failed! Dimension doesn't fit!")
    
    
def increase_dims(data):
    _,dim0=data.shape
    k=0
    lookup_class=dict()
    for i in range(dim0-1):
        for j in range(i+1,dim0):
            lookup_class[k]=[i,j]
            k += 1
    result=np.hstack([data]+[(data[:,lookup_class[i][0]]*data[:,lookup_class[i][1]]).reshape(-1,1) for i in range(dim0*(dim0-1)//2)]+[data[:,i].reshape(-1,1)**2 for i in range(dim0)])
    return result/np.max(result)
