import time, os, sys
import ctypes, math
import numpy, scipy
import operator
#import statistics
#from statistics import mean, median, variance, stdev
import sklearn

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from scipy.signal import hann
from scipy.fftpack import rfft
from scipy.signal import butter, lfilter, freqz,lp2hp
from numpy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline
import operator
from read_data import *
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA


import cmath
from cmath import *

# visualization T-SNE

def visualize_SNE(X,y,per,iter,n):
    N = 10000
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    
    rndperm = numpy.random.permutation(df.shape[0])
    
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values
    
    #pca = PCA(n_components=2)
    #pca_result = pca.fit_transform(data_subset)
    #df_subset['pca-one'] = pca_result[:,0]
    #df_subset['pca-two'] = pca_result[:,1] 
    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=per, n_iter=iter)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure()
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", n),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()

# normalization with PCA whitening


def zca_whitening_matrix(X):

    sigma = numpy.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = numpy.linalg.svd(sigma)
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = numpy.dot(U, numpy.dot(numpy.diag(1.0/numpy.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

#normalization, standarization, scalarization

def normalize_data(data,norm):  
  if norm =='scaling':
    #scaler = StandardScaler().fit(data) #standarize
    #datasc = scaler.transform(data)
    scaler2 = MinMaxScaler(feature_range=(0, 1)) #rescale
    data_norm = scaler2.fit_transform(data) 
    plt.pause(1)
  elif norm =='zca':
    ZCAMatrix = zca_whitening_matrix(data) # get ZCAMatrix
    data_norm = numpy.dot(ZCAMatrix, data)
  return data_norm

# interploation:

def max_size(list):
  matlen=[]
  for k in range(0, len(list)): # return the maximum size
    lenght=len(list[k])
    matlen.append(lenght)
    
  index, value = max(enumerate(matlen), key=operator.itemgetter(1))
  value1=int(numpy.mean(matlen))

  #value1=int(mean(matlen))
  return value1

def interpolate_data(value, data):
  #data=map(int,data)
  mat=[]  
  for k in range(0, len(data)):
    old_indices = numpy.arange(0,len(data[k]))
    new_length = value
    new_indices = numpy.linspace(0,len(data[k])-1,new_length)
    spl = UnivariateSpline(old_indices,data[k],k=3,s=0)
    new_array = spl(new_indices)
    mat.append(new_array)
  return mat

# filtering:

def get_fft_values(y_values, T, N, f_s):
    fft_values_ = scipy.fftpack.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return  fft_values

def time_freq(data):
  dataf=[]
  for x in xrange(0,len(data)):
    vect=data[x]
    freq=scipy.fftpack.fft(vect)
    #print("fft",freq)
    #plt.plot(freq)
    #plt.show()
    #freq = get_fft_values(y_values, T, N, f_s)
    dataf.append(freq)
  return dataf


def norm_complex(data):
  vec1mat=[]
  for x in xrange(0,len(data)):
    vec=data[x]
    vec1array=[]
    for j in xrange(0,len(vec)):
      vec1=abs(vec[j]) # real part of the complex value
      vec1array.append(vec1)
    vec1mat.append(vec1array)
  return vec1mat

def process_vect(data): 
  #w = savgol_filter(data[k], 101, 2)
  n = 20  # the larger n is, the smoother curve will be
  b = [1.0 / n] * n
  a = 1
  yy = lfilter(b,a,y)
  return yy


def conc_data(data1,data2):
  dataf=[]
  for x in xrange(0,len(data1)):
    vec1 = data1[x]
    vec2 = data2[x]
    vec = data1 + data2
    dataf.append(vec)
  return dataf

def read_prepare_data(path1,path2,mod,norm): # normalization type: 'scaling','zca', representation: 'freq','spectr',time

  X1veh,X1acc,Y1,Y2,Y3=read_data(path1,mod) #  get the data and labels from veh files
  Xgyro,Yd,Ygen  = read_grodata(path2)

  #X1veh,X1acc,accx,accy,accz,Y1,Y2,Y3=read_data(path,mod) #  get the data and labels from veh files

  mean1 = max_size(X1veh)
  data_inter_veh = interpolate_data(mean1,X1veh)  # interpolate data samples to the mean size
  
  mean2 = max_size(X1acc)
  data_inter_acc = interpolate_data(mean1,X1acc)

  mean3 = max_size(Xgyro)
  data_inter_gyro = interpolate_data(mean3,Xgyro)
  '''
  meanx = max_size(accx)
  inter_accx = interpolate_data(meanx,accx)
  meany = max_size(accy)
  inter_accy = interpolate_data(meany,accy)
  meanz = max_size(accz)
  inter_accz = interpolate_data(meanz,accz)
  '''
  #data_fil = process_vect(data_inter)

  data_veh_time = normalize_data(data_inter_veh,norm) # veh
  data_acc_time = normalize_data(data_inter_acc,norm) #acc
  data_gyro_time = normalize_data(data_inter_gyro,norm) #acc

  #data_accx_time = normalize_data(inter_accx,norm) #acc
  #data_accy_time = normalize_data(inter_accy,norm) #acc
  #data_accz_time = normalize_data(inter_accz,norm) #acc

  datafreq1 = time_freq(data_veh_time)
  data_veh_freq = norm_complex(datafreq1)
  data_veh_freq = normalize_data(data_veh_freq,'scaling')

  datafreq2 = time_freq(data_acc_time)
  data_acc_freq = norm_complex(datafreq2)
  data_acc_freq = normalize_data(data_acc_freq,'scaling')
  

  datafreq3 = time_freq(data_gyro_time)
  data_gyro_freq = norm_complex(datafreq3)
  data_gyro_freq = normalize_data(data_gyro_freq,'scaling')

  #vehtime,vehfreq,acctime,acctimex,acctimey,acctimez,accfreq,YG,Yhot,YS= shuffle(data_veh_time,data_veh_freq,data_acc_time,data_acc_timex,data_acc_timey,data_acc_timez,data_acc_freq,Y1,Y2,Y3)  # shuffeling
  vehtime,vehfreq,acctime,accfreq,YG,Yhot,YS= shuffle(data_veh_time,data_veh_freq,data_acc_time,data_acc_freq,Y1,Y2,Y3)  # shuffeling
  gyrotime,gyrofreq,Ygyrod,Ygyrog= shuffle(data_gyro_time,data_gyro_freq,Yd,Ygen)  # shuffeling

  return vehtime,vehfreq,acctime,accfreq,gyrotime,gyrofreq,YG,Yhot,YS,Ygyrod,Ygyrog
  #return vehtime,vehfreq,acctime,acctimex,acctimey,acctimez,accfreq,YG,Yhot,YS

