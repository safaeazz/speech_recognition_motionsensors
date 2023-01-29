from deepautoencoder_012 import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io as sio
##import pandas
import scipy
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
##import plotly.plotly as py
##import pylab as plt
import tensorflow as tf

import  skopt
from  skopt import  gp_minimize , forest_minimize


from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
##from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tempfile import TemporaryFile
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


dim_learning_rate = Categorical(categories=['0.0005'], name='learning_rate')
dim_learning_rate2 = Categorical(categories=['0.001'], name='learning_rate2') 
dim_num_dense_nodes1 = Categorical(categories=['100','600','1200'], name='num_dense_nodes1')   #,'1500','2000'
dim_alpha = Categorical(categories=['0.0001','0'], name='alpha') 
dim_beta = Categorical(categories=['100','10','1','0.0001','0'], name='beta') #,,'0.1','0.01','0.001','0.0001','0.00001','0.000001'    '1','2','3','4','5','6','7','8','9','10','100'    '0.001','0.01',  ,'0.5','2.5','7.5'  ,'0.5','0.1','0.01','0.001','0.00001','0'
dim_acf = Categorical(categories=['sigmoid','tanh'], name='acf') #,'relu',,'tanh' 
dim_dcf = Categorical(categories=[ 'sigmoid'], name='dcf') #,'linear'
dim_winit = Categorical(categories=['2'], name='winit') #'1','2','3','4','5','6','7'
dim_w = Categorical(categories=['tied','transpose'], name='w') #'he_uniform','caffe_uniform','he_normal','caffe_normal'
dim_opt = Categorical(categories=['Adam'] ,name='opt') #  'Adam',  ,'GradientDescent' ,'Adadelta','Adagrad' 'RMSProp',
dim_epoch=Categorical(categories=['2000'], name='epoch')   #,'3000','5000'
dim_L = Categorical(categories=['rmse_LDA1','rmse_LDA2','rmse_LDA1_TFT','rmse_LDA2_TFT','rmse_LDA1_S1','rmse_LDA1_S2','rmse_LDA1_S3','rmse_LDA1_S4'], name='L')
##dim_Drop = Categorical(categories=['0.01','0.05','0.1','0.15','0.2','0.25'], name='Drop')
dim_stdr = Categorical(categories=['0'], name='std')
dim_HC = Categorical(categories=['1'], name='HC') # ,'3'
dim_Coef = Categorical(categories=['0.5'], name='Coef') #,'0.5','0.75','1'   ,'0.5','0.75'
dim_epochFC=Categorical(categories=['20000'], name='epochFC') #'500', '1000', '3000','6000', '9000', '12000','15000'
dim_C = Categorical(categories=['0.01'], name='C')  #,'0.1','1'
dim_G = Categorical(categories=['0.001'], name='G')  #'0.01','0.1','1','10'
dim_batch= Categorical(categories=['50'], name='batch') #



dimensions = [dim_learning_rate,dim_learning_rate2,
              dim_num_dense_nodes1,
              dim_alpha,
              dim_beta,
              dim_acf,
              dim_dcf,
              dim_winit,
              dim_opt,
              dim_epoch,dim_HC,dim_Coef,dim_C,dim_G,dim_batch,dim_stdr,dim_w,dim_L]  #,dim_stdr,dim_epochFC,dim_Drop


default_parameters = ['0.0005', '0.001', '100', '0', '1', 'tanh', 'sigmoid', '2', 'Adam', '2000', '1', '0.5', '0.01', '0.001','50','0','transpose','rmse_LDA1_S1']


def log_dir_name(learning_rate,learning_rate2,num_dense_nodes1,alpha,beta,acf,dcf,winit,opt,epoch,HC,Coef,C,G,batch,std,w,L):  

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_nodes_{1}_{2}/"

    # Insert all the hyper-parameters in the dir-name.(
    log_dir = s.format(learning_rate,learning_rate2,
                       num_dense_nodes1,
                       alpha,
                       beta,acf,dcf,winit,opt,epoch,HC,Coef,C,G,batch,std,w,L) 

    return log_dir

best_c=0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate,learning_rate2,num_dense_nodes1 ,alpha,beta,acf,dcf,winit,opt,epoch,HC,Coef,C,G,batch,std,w,L): 

    # Print the hyper-parameters.
    print('learning rate: ',learning_rate)
##    print('learning rate2: ',learning_rate2)
    print('num_dense_nodes1: ', num_dense_nodes1)
    print('alpha: ',alpha)
    print('beta: ',beta)
    print('acf: ',acf)
    print('dcf: ',dcf)
##    print('winit',winit)
    print('opt',opt)
    print('Epoch DRAE ',epoch)
##    print('stdr :', std)
##    print('HC :',HC)
##    print('Coef. :',Coef)
##    print('C :',C)
##    print('G. :',G)    
##    print('Batch  :',batch)
    print('w. :',w)    
    print('loss  :',L)
##    print('Epoch FC :',epochFC)
    print()

    mat = sio.loadmat('R2_sort_G0D_G2A_mat2ray_adjust(img)_pas10.mat', squeeze_me=True)  

    X0=mat['R10']  
    X2=mat['R12']

##    XE=mat['R122_v']

    X0=X0[0:1300]
    X2=X2[0:1300]

    y0=np.zeros(len(X0))
    y2=np.ones(len(X2))            
    y=np.concatenate((y0,y2),axis=0)

    
    X=np.concatenate((X0, X2), axis=0)

    if(std==1):
        s = StandardScaler().fit(X)
        X0 = s.transform(X0)
        X2 = s.transform(X2)
        
    n=2
    skf = StratifiedKFold(n_splits=n,shuffle=True)

##    kf = KFold(n_splits=int(n),shuffle=True) 
    f=fm=0
    atr=ats=0
    for train_index, test_index in skf.split(X, y):
        f+=1
        print("||||||||||||||||||||||||||||||||||||||||||||")
        print("------------------------------------- Data",f)
        print("||||||||||||||||||||||||||||||||||||||||||||")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X0_train=X_train[y_train==0]
        X2_train=X_train[y_train==1]
        X0_test=X_test[y_test==0]
        X2_test=X_test[y_test==1]
##        X0_train, X0_test = X0[train_index], X0[test_index]
##        X2_train, X2_test = X2[train_index], X2[test_index]
        XE=np.concatenate((X0_train, X2_train), axis=0)

        x0=X0_train
        x1=X2_train
        x0vl=X0_test
        x1vl=X2_test
        
        model1 = StackedAutoEncoder(dims=[int(num_dense_nodes1)], act=acf,dct=dcf, loss=L, lr=float(learning_rate),W=w,
                                   batch_size=int(batch), print_step=51,alpha=float(alpha),beta=float(beta),winit=winit,opt=opt , epoch=int(epoch),C=float(C),G=float(G)) #, HC=int(HC),Coef=float(Coef)

        F0,F0t,F2,F2t,W1,b1,f1,ats,pts,rts,fts,Lts,Lt=model1.fit(X0_train,X2_train,X0_test,X2_test,XE,1)  
        sio.savemat('Features_R1_fold_'+str(f)+'_AE2.mat',{'F0': F0, 'F2': F2, 'F0t': F0t, 'F2t': F2t})
        
##        if f1<0.5:
##            break
        fm=fm+f1
        
    c=fm/n
    print("====================================")
    print("Score: {0:.2%}".format(c))
    print("====================================")

    global best_c
    
    if c > best_c:
        # Update the classification accuracy.
        best_c = c #
        
        sio.savemat('R1.mat',{'F0': F0, 'F2': F2, 'F0t': F0t, 'F2t': F2t})

    return -c


search_result  = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=100,
                            x0=default_parameters)
print(search_result.x)