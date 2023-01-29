#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from functions import *
from read_data import *
from data_preprocessing import *
from classification import *
from autoencoder import *

# scatter plot of blobs dataset
from numpy import where
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from tabulate import tabulate
import pandas as pd
from plotstft import *
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -
# read and prepare the data and labels from files
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -

#path =".../VEH/VEH Dataset/"

path1 = "/home/safae/Documents/phd/vehproject/VEH_DATASET/VEH_Dataset/"
path2 =    "/home/safae/Documents/phd/gyro/gyro-audio/"
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - 
# get the parameters from terminal
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task', 'digits', 'data task') 
flags.DEFINE_string('norm', 'scaling', 'data normalization') 
flags.DEFINE_integer('mod', '1', 'data modality')

flags.DEFINE_boolean('noise', False, 'noise') 
flags.DEFINE_float('ratio', 0.2, 'noise ratio') 

flags.DEFINE_boolean('sparse', False, 'sparsity') 
flags.DEFINE_float('beta', 0.1, 'brta') 
flags.DEFINE_float('lamda', 0.2, 'lamda') 


flags.DEFINE_integer('h1', 1200, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h2', 1000, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h3', 800, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h4', 512, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h5', 256, 'hidden layer size') # hidden layer size



flags.DEFINE_integer('ep1', 30, 'epochs') # batch size
#flags.DEFINE_integer('ep2', 100, 'epochs') # batch size
flags.DEFINE_integer('ep', 1000, 'epochs') # batch size


flags.DEFINE_float('lr1', 0.001, 'learning_rate') # learning rate
#flags.DEFINE_float('lr2', 0.001, 'learning_rate') # learning rate

flags.DEFINE_float('momentum', 0.7, 'momentum') # momentum

flags.DEFINE_integer('bsize1', 30, 'batch size') # batch size
flags.DEFINE_integer('bsize2', 30, 'batch size') # batch size
flags.DEFINE_integer('bsize', 30, 'batch size') # batch size


flags.DEFINE_string('act', 'sigmoid', 'activation function') #tanh, relu, selu,elu,linear
flags.DEFINE_string('we', 'xavier_uniform', 'weight init') #'xavier_uniform','xavier_normal','he_normal','he_uniform','caffe_uniform'
flags.DEFINE_string('loss', 'mse', 'loss function') #cross_entropy, mse
flags.DEFINE_string('opt', 'Rmsprop', 'optimizer') # momentum,ada_grad, gradient_descent

#data_name = FLAGS.data
task = FLAGS.task
norm = FLAGS.norm # for early fusion
mod = FLAGS.mod
noise = FLAGS.noise
ratio = FLAGS.ratio
sparse = FLAGS.sparse
beta = FLAGS.beta
lamda = FLAGS.lamda

dim1 = FLAGS.h1
dim2 = FLAGS.h2
dim3 = FLAGS.h3
dim4 = FLAGS.h4
dim5 = FLAGS.h5

ep1 = FLAGS.ep1
#ep2 = FLAGS.ep2
ep = FLAGS.ep

lr1 = FLAGS.lr1
#lr2 = FLAGS.lr2

batch1 = FLAGS.bsize1
batchf = FLAGS.bsize2
batch = FLAGS.bsize

batch3 = batch4 = batch5 = batch2 = batch1
lr3 = lr4 = lr5 = lr2 = lr1
ep3 = ep4 = ep5 = ep2 = ep1

weight_init = FLAGS.we   
loss_func = FLAGS.loss
opt = FLAGS.opt
act_func=FLAGS.act
momentum = FLAGS.momentum


print("==============================================================")
print (" ------- STEP0: read & prepare the data ---------------------")
print("===============================================================")

vehtime,vehfreq,acctime,accfreq,gyrotime,gyrofreq,YG,Yhot,YS,Ygyro1,Ygyro2 = read_prepare_data(path1,path2,mod,norm)

if task == 'gender':
   labels = YG
   n_classes = 2
elif task == 'hot_or_not':
   labels = Yhot
   n_classes = 2
elif task == 'sentences':
   labels = YS
   n_classes = 4
elif task == 'digits':
   labels = Ygyro1
   n_classes = 11
elif task == 'gen':
   n_classes=2
tr_X1, te_X1,tr_X2, te_X2, tr_Y1, te_Y1 = train_test_split(vehtime,vehfreq, labels, test_size=0.2, random_state=42)
#tr_X1, te_X1,tr_X2, te_X2, tr_Y1, te_Y1 = train_test_split(gyrotime,gyrofreq, Ygyro1, test_size=0.3, random_state=42)

#tr_X1 = gyrotime
#tr_X2 = gyrofreq
#
#tr_Y1 = Ygyro2

print("step 0 done...")

'''
print("==============================================================")
print (" ------------------ visualization T-SNE ---------------------")
print("===============================================================")
'''

#plotstft(vehtime)
vt = len(vehtime[0])
vf = len(vehfreq[0])

#vt = len(vehtime[0])
#vf = len(vehfreq[0])
#acct = len(acctime[0])
#accf = len(accfreq[0])

#vt = len(gyrotime[0])
#vf = len(gyrofreq[0])

samples = len(tr_X1)
print(vt,vf)

print("==============================================================")
print (" ------- STEP1: learning features ---------------------")
print("===============================================================")

print ("Modality 1: VEH data")
print (" start training ......")
print("selected parameters")

print(tabulate([['hid1',dim1],['hid2',dim2],['epochs',ep1],["learning_rate",lr1],["batch size",batch1],["optimizer",opt],["noise",noise],["ratio",ratio],["sparsity",sparse],["beta",beta],["lambda",lamda]],tablefmt='fancy_grid'))

f_veht1,wvt1,bvt1 = run_session(tr_X1,noise,ratio,sparse,beta,lamda,vt,dim1,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvt1en","wvt1dec","bvt1en","bvt1dec")

f_veht1 = normalize_data(f_veht1,norm)


f_veht2,wvt2,bvt2 = run_session(f_veht1,noise,ratio,sparse,beta,lamda,dim1,dim2,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvt2en","wvt2dec","bvt2en","bvt2dec")
f_veht2 = normalize_data(f_veht2,norm)


#get_cross_validation(f_veht2,tr_Y1)

f_vehf1,wvf1,bvf1 = run_session(tr_X2,noise,ratio,sparse,beta,lamda,vf,dim1,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvf1en","wvf1dec","bvf1en","bvf1dec")
f_vehf1 = normalize_data(f_vehf1,norm)

#get_cross_validation(f_vehf1,tr_Y1)

f_vehf2,wvf2,bvf2 = run_session(f_vehf1,noise,ratio,sparse,beta,lamda,dim1,dim2,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvf2en","wvf2dec","bvf2en","bvf2dec")
f_vehf2 = normalize_data(f_vehf2,norm)
#get_cross_validation(f_vehf2,tr_Y1)


print (" training join representation ...")

#veh_freq_time0=np.multiply(f_veht2, f_vehf2)
veh_freq_time = np.concatenate((f_veht2,f_vehf2),axis=1)	
veh_freq_time = normalize_data(veh_freq_time,norm)
numv = len(veh_freq_time[0])
print("numv",numv)

f_veh_ft1,wv_ft1,bv_ft1 = run_session(veh_freq_time,noise,ratio,False,3,3e-3,numv,dim2,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvft1en","wvft1dec","bvft1en","bvft1dec")

f_veh_ft1 = normalize_data(f_veh_ft1,norm)

get_cross_validation(f_veh_ft1,tr_Y1)

#f_veh_ft2,wv_ft2,bv_ft2 = run_session(f_veh_ft1,noise,ratio,False,3,3e-3,dim1,dim2,act_func,lr1,momentum,weight_init,
#                                   loss_func,opt,batch1,ep1,"wvft2en","wvft2dec","bvft2en","bvft2dec")
#
#f_veh_ft2 = normalize_data(f_veh_ft2,norm)
#
#get_cross_validation(f_veh_ft2,tr_Y1)

print("================================================================")
print (" ------- STEP2: Supervised classification ---------------------")
print("=================================================================")

# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -
#  finetuning for a stacked AE and softmax classification
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -

epsilon = 1e-3

def batch_norm(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training == True:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


print (" classification using  softmax ===== ...")

tf.reset_default_graph()

X1 = tf.placeholder("float", [None, vt])
X2 = tf.placeholder("float", [None, vf])
Y1 = tf.placeholder(tf.int32, [None, n_classes], name='Y_placeholder')
learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
is_training = tf.placeholder(tf.bool, name="is_train")
prob = tf.placeholder_with_default(1.0, shape=())

w11n = tf.get_variable("weight-time1",  dtype=tf.float32,initializer=wvt1)
w12n = tf.get_variable("weight-time2",  dtype=tf.float32,initializer=wvt2)

w21n = tf.get_variable("weight-freq1",  dtype=tf.float32,initializer=wvf1)
w22n = tf.get_variable("weight-freq2",  dtype=tf.float32,initializer=wvf2)

b11n = tf.get_variable("bias-time1",  dtype=tf.float32,initializer=bvt1)
b12n = tf.get_variable("bias-time2",  dtype=tf.float32,initializer=bvt2)

b21n = tf.get_variable("bias-freq1",  dtype=tf.float32,initializer=bvf1)
b22n = tf.get_variable("bias-freq2",  dtype=tf.float32,initializer=bvf2)

w3n = tf.get_variable("weight-timefreq1",  dtype=tf.float32,initializer=wv_ft1)
b3n = tf.get_variable("bias-timefreq1",  dtype=tf.float32,initializer=bv_ft1)

#w4n = tf.get_variable("weight-time-feq2",  dtype=tf.float32,initializer=wv_ft2)
#b4n = tf.get_variable("bias-time-feq2",  dtype=tf.float32,initializer=bv_ft2)

w4n = get_weight_variable([dim2, dim3], 'w4', 'xavier_uniform')
b4n = tf.Variable(tf.random_normal(([dim3]), name="b4"))

layer11 = tf.nn.sigmoid(tf.add(tf.matmul(X1, w11n),b11n)) 
BN01 = tf.nn.dropout(batch_norm(layer11, is_training, decay = 0.999), prob)
#BN01_drop = tf.layers.dropout(BN01, r, training=is_training)

layer12 = tf.nn.sigmoid(tf.add(tf.matmul(BN01, w12n),b12n)) 
BN1 = tf.nn.dropout(batch_norm(layer12, is_training, decay = 0.999), prob)


layer21 = tf.nn.sigmoid(tf.add(tf.matmul(X2, w21n),b21n)) 
BN02 = tf.nn.dropout(batch_norm(layer21, is_training, decay = 0.999), prob)

layer22 = tf.nn.sigmoid(tf.add(tf.matmul(BN02, w22n),b22n)) 
BN2 = tf.nn.dropout(batch_norm(layer22, is_training, decay = 0.999),prob)

layerconcat = tf.concat([BN1,BN2],axis=1)
#layerconcat = tf.multiply(BN1,BN2)
BNc  = tf.nn.dropout(batch_norm(layerconcat, is_training, decay = 0.999),prob)


layer3 = tf.nn.sigmoid(tf.add(tf.matmul(BNc, w3n),b3n))
l3_BN = tf.nn.dropout(batch_norm(layer3, is_training, decay = 0.999),prob)

#l3_BN = batch_norm(layer3, is_training, decay = 0.999)

#l3_drop = tf.layers.dropout(l3_BN, r, training=is_training)

layer4 = tf.nn.sigmoid(tf.add(tf.matmul(l3_BN, w4n),b4n))
l4_BN = batch_norm(layer4, is_training, decay = 0.999)


### output layer softmax

w = get_weight_variable([dim3, n_classes], 'w-logits', 'xavier_uniform')
b = tf.Variable(tf.random_normal(([n_classes]), name="bias"))

# this logits will be later passed through softmax layer
logits = tf.matmul(l4_BN, w) + b# Define loss and optimizer
# Define loss and optimizer

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y1))
#loss2 = 0.001* (tf.nn.l2_loss(w11n) + tf.nn.l2_loss(w12n) + tf.nn.l2_loss(w21n) + tf.nn.l2_loss(w22n) + tf.nn.l2_loss(w3n) + tf.nn.l2_loss(w)) 

#tf.contrib.layers.l1_l2_regularizer(
#    scale_l1=1.0,
#    scale_l2=1.0,
#    scope=None
#)

values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
alpha = values[3]

vars   = tf.trainable_variables() 
l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=alpha,scale_l2=alpha, scope=None)
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=alpha,scope=None)
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=alpha,scope=None)

regl1 = tf.contrib.layers.apply_regularization(l1_regularizer, vars)
regl2 = tf.contrib.layers.apply_regularization(l2_regularizer, vars)
regl12 = tf.contrib.layers.apply_regularization(l1_l2_regularizer, vars)


#reg2 = tf.add_n([tf.nn.l2_loss(v) for v in vars ]) * alpha
#l2_loss
#l1 = 0.001*tf.reduce_sum(tf.abs(parameters))

loss_op = loss1 

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss_op)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=1.0).minimize(loss1)
#AdamOptimizer
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# encode labels to one hot 
# split the data into train and test

# run a session
init = tf.global_variables_initializer()
#init = tf.initialize_variables([w,b])
tr_Y = one_hot(tr_Y1)
te_Y = one_hot(te_Y1)
# paraneters to optimize: batch,epochs,learning rate
eptab=[]
trloss=[]
teloss=[]
teacc=[]
Ws = []
acct,l1,l2 = [],[],[]

with tf.Session() as sess:
    writer = tf.summary.FileWriter('graph', sess.graph)
    sess.run(init)	
    #print(w1n.eval())
    plt.figure()	
    for k in range(0,251): # for all epochs    

        avg_cost = 0
        total_batch = int(len(tr_X1)/batchf)
        # Loop over all batches        

        for j in range (0,tr_X1.shape[0], batchf):
           
            if k < 251:
                lr = 0.0001 
            elif 51 < k < 100:
                lr = 0.00001

            batch_x1 = tr_X1[j:j+batchf]
            batch_x2 = tr_X2[j:j+batchf]
            batch_y = tr_Y[j:j+batchf]
            _, l = sess.run([optimizer, loss_op], feed_dict={X1: batch_x1,
            	                                             X2: batch_x2,
	                                                         Y1: batch_y,
                                                            is_training: True,
                                                            learning_rate_placeholder:lr,
                                                            prob:0.25
                                                            })
            avg_cost += l / total_batch
            #print(sess.run([w4n[0]]))

        #if k % 20 == 0:
        err,acc = sess.run([loss_op,accuracy], feed_dict={X1:te_X1,
        									X2:te_X2,
        									Y1:te_Y,
        									is_training: False})
        l3dis = sess.run([l3_BN],feed_dict={X1: batch_x1,X2: batch_x2,Y1: batch_y,is_training: True})
        print("Epoch:", k, "training loss = ",avg_cost,"test loss=",err,"test acc =",acc)
        sns.distplot((np.array(l3dis)))
        #print("after 20 epochs =",w1n.eval())
        eptab.append(k)
        trloss.append(avg_cost)
        teloss.append(err)
        teacc.append(acc)
    plt.show()
    print("Optimization Finished!")
    l1tr,l2tr,l3tr,l4tr = sess.run([BN1,BN2,l3_BN,l4_BN],feed_dict={X1:tr_X1,
        									X2:tr_X2,
        									Y1:tr_Y,
        									is_training: False})

    l1te,l2te,l3te,l4te = sess.run([BN1,BN2,l3_BN,l4_BN],feed_dict={X1:te_X1,
        									X2:te_X2,
        									Y1:te_Y,
        									is_training: False})


#print("acc" ,acctest)
plt.plot(eptab,trloss,color='red')
plt.plot(eptab,teloss,color='blue')
plt.plot(eptab,teacc,color='green')
plt.show()

L1tr = np.array(l1tr)
L2tr = np.array(l2tr)
L3tr = np.array(l3tr)
Logtr = np.array(l4tr)

L1te = np.array(l1te)
L2te = np.array(l2te)
L3te = np.array(l3te)
Logte = np.array(l4te)

print("SVM")

clf = SVC(gamma=0.01, C=100)
clf.fit(Logtr, tr_Y1)
ypred = clf.predict(Logte)
print("accuracy", accuracy_score(te_Y1,ypred))

print("visualize training")
visualize_SNE(L1tr,tr_Y1,30,1000,n_classes)
print("visualize training")
visualize_SNE(L2tr,tr_Y1,30,1000,n_classes)
print("visualize training")
visualize_SNE(L3tr,tr_Y1,30,1000,n_classes)
print("visualize training")
visualize_SNE(Logtr,tr_Y1,30,1000,n_classes)


print("visualize testing1")
visualize_SNE(L1te,te_Y1,30,1000,n_classes)
print("visualize testing2")
visualize_SNE(L2te,te_Y1,30,1000,n_classes)
print("visualize testing3")
visualize_SNE(L3te,te_Y1,30,1000,n_classes)
visualize_SNE(Logte,te_Y1,30,1000,n_classes)
