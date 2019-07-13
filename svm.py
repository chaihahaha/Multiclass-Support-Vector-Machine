import tensorflow as tf
import numpy as np
import operator as op
from functools import reduce

class svm_model:
    def __init__(self,n_class, dimension):
        self.n_class=n_class
        self.dimension=dimension
        self.w=[]
        self.b=[]
        self.logit=[]
        self.logit_tmp=[]
        self.loss=[]
        #prediction=[]
        self.correct_logit=[]
        self.lookup_class=dict()
        self.w_model=np.random.uniform(0,1,(self.__ncr(n_class,2), dimension)).astype(np.float32)
        self.b_model=np.random.uniform(0,1,(self.__ncr(n_class,2),1)).astype(np.float32)
        self.lookup_matrix=np.zeros((n_class, self.__ncr(n_class,2)),dtype=np.float32)
        self.batch_x=tf.placeholder(tf.float32,shape=(None,dimension),name="batch_x")
        self.batch_y=tf.placeholder(tf.float32,shape=(None,1),name="batch_y")
        self.w = tf.Variable(tf.random_uniform([self.__ncr(n_class,2), self.dimension]))
        self.b = tf.Variable(tf.random_uniform([self.__ncr(n_class,2),1]))
        k=0
        for i in range(n_class-1):
            for j in range(i+1,n_class):
                self.lookup_class[k]=[i,j]
                k += 1

        for i in range(n_class):
            for j in range(self.__ncr(n_class,2)):
                if i in self.lookup_class[j]:
                    if self.lookup_class[j][0]==i:
                        self.lookup_matrix[i,j]=1.0
                    else:
                        self.lookup_matrix[i,j]=-1.0

        
        for i in range(self.__ncr(n_class,2)):
            idx=tf.where(tf.keras.backend.any(tf.equal(self.batch_y,self.lookup_class[i]),1)) # tf.where and tf.gather_nd is equivalent to a[condition is true]
            self.logit.append(tf.matmul(tf.reshape(self.w[i,:],(1,dimension)), tf.gather_nd(self.batch_x, idx), transpose_b=True) + self.b[i,:])
            self.logit_tmp.append(tf.tanh(self.logit[i]))
            self.correct_logit.append(self.__zonp(tf.cast(tf.equal(tf.gather_nd(self.batch_y, idx),self.lookup_class[i][0]),tf.float32)))
            self.loss.append(tf.maximum(0.0,1-tf.matmul(self.logit_tmp[i],self.correct_logit[i])) + tf.norm(self.w[i,:], 2))

        self.prediction = tf.argmax(tf.matmul(self.lookup_matrix, tf.tanh(tf.matmul(self.w, self.batch_x, transpose_b=True) + self.b)),axis=0)
        self.total_loss = sum(self.loss)
        self.opt = tf.train.RMSPropOptimizer(1e-2).minimize(self.total_loss)
        
    def __ncr(self,n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom
    def __zonp(self,zero_one):
        return 2*zero_one-1

    def fit(self,data_x,data_y,iter_time=1000):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([tf.assign(self.w,self.w_model),tf.assign(self.b,self.b_model)])
            for _ in range(iter_time):
                _,loss_val,w_model,b_model = sess.run([self.opt,self.total_loss,self.w,self.b], feed_dict={self.batch_x:data_x,self.batch_y:data_y})
                print(loss_val)
        self.w_model = w_model
        self.b_model = b_model
        return w_model,b_model
    def predict(self,data_x):
        with tf.Session() as sess:  
            result = sess.run(self.prediction,feed_dict={self.batch_x:data_x,self.w:self.w_model,self.b:self.b_model})
        return result.reshape(-1,1)
def increase_dims(data):
    _,dim0=data.shape
    k=0
    lookup_class=dict()
    for i in range(dim0-1):
        for j in range(i+1,dim0):
            lookup_class[k]=[i,j]
            k += 1
    #print([data]+[data[:,lookup_class[i][0]]*data[:,lookup_class[i][1]] for i in range(dim0*(dim0-1)//2)])
    result=np.hstack([data]+[(data[:,lookup_class[i][0]]*data[:,lookup_class[i][1]]).reshape(-1,1) for i in range(dim0*(dim0-1)//2)])
    return result
