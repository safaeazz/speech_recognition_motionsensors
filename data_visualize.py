import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from read_data import *
import data_preprocessing
from data_preprocessing import *
import librosa
from librosa import display

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

def time_freq(data):
  dataf=[]
  for x in xrange(0,len(data)):
    vect=data[x]
    freq=scipy.fftpack.fft(vect)
    dataf.append(freq)
  return dataf

import cmath
from cmath import *

def norm_complex(data):
  vec1mat=[]
  for x in xrange(0,len(data)):
    vec=data[x]
    vec1array=[]
    for j in xrange(0,len(vec)):
      vec1=abs(vec[j])
      vec1array.append(vec1)
    vec1mat.append(vec1array)
  return vec1mat

def process_vect(data): 
  mat=[]
  for k in range(0, len(data)):
    #w = savgol_filter(data[k], 101, 2)
    n = 20  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b,a,data[k])
    #mat1.append(yy)

    mat.append(yy)
  return mat


def conc_data(data1,data2):
  dataf=[]
  for x in xrange(0,len(data1)):
    vec1 = data1[x]
    vec2 = data2[x]
    vec = data1 + data2
    dataf.append(vec)
  return dataf

def read_prepare_data(path,mod,norm): # normalization type: 'scaling','zca', representation: 'freq','spectr',time

  X1veh,X1acc,accx,accy,accz,Y1,Y2,Y3=read_data(path,mod) #  get the data and labels from veh files

  mean1 = max_size(X1veh)
  data_inter_veh = interpolate_data(mean1,X1veh)  # interpolate data samples to the mean size
  
  mean2 = max_size(X1acc)
  data_inter_acc = interpolate_data(mean1,X1acc)

  meanx = max_size(accx)
  inter_accx = interpolate_data(meanx,accx)
  meany = max_size(accy)
  inter_accy = interpolate_data(meany,accy)
  meanz = max_size(accz)
  inter_accz = interpolate_data(meanz,accz)
  
  #data_fil = process_vect(data_inter)

  data_veh_time = normalize_data(data_inter_veh,norm) # veh
  data_acc_time = normalize_data(data_inter_acc,norm) #acc
  accx2 = normalize_data(inter_accx,norm) #acc
  accy2 = normalize_data(inter_accy,norm) #acc
  accz2 = normalize_data(inter_accz,norm) #acc


  vehtime,acctime,acctimex,acctimey,acctimez,YG,Yhot,YS= shuffle(X1veh,data_acc_time,accx,accy,accz,Y1,Y2,Y3)  # shuffeling


  return vehtime,acctime,acctimex,acctimey,acctimez,YG,Yhot,YS



path1 = "/home/safae/Documents/phd/vehproject/VEH_DATASET/VEH_Dataset/"
'''
X1v,X1a,Y1,Y2,Y3=read_data(path,1) #  get the data and labels from veh files
X1veh,X1acc,Yg,Yh,Ys = shuffle(X1v,X1a,Y1,Y2,Y3)
mean1 = max_size(X1acc)
data_inter_veh = interpolate_data(mean1,X1acc)  # interpolate data samples to the mean size
data_veh_time = normalize_data(data_inter_veh,'scaling') # veh
'''

vehtime,acctime,acc_timex,acc_timey,acc_timez,Yg,Yh,Ys = read_prepare_data(path1,path2,'scaling')

maleokx=[]
maleoky=[]
maleokz=[]

femaleokx=[]
femaleoky=[]
femaleokz=[]

males1x=[]
males1y=[]
males1z=[]

females1x=[]
females1y=[]
females1z=[]

males2x=[]
males2y=[]
males2z=[]

females2x=[]
females2y=[]
females2z=[]

males3x=[]
males3y=[]
males3z=[]

females3x=[]
females3y=[]
females3z=[]
'''
for i in xrange(0,len(Yg)):
  if Yg[i] == 0:
    if Ys[i] == 0: 
      maleokx.append(acc_timex[i])
      maleoky.append(acc_timey[i])
      maleokz.append(acc_timez[i])
    elif Ys[i] == 1:
      males1x.append(acc_timex[i])
      males1y.append(acc_timey[i])
      males1z.append(acc_timez[i])
    elif Ys[i] == 2:
      males2x.append(acc_timex[i])
      males2y.append(acc_timey[i])
      males2z.append(acc_timez[i])
    elif Ys[i] == 3:
      males3x.append(acc_timex[i])
      males3y.append(acc_timey[i])
      males3z.append(acc_timez[i])
  elif Yg[i] == 1:
    if Ys[i] == 0: 
      femaleokx.append(acc_timex[i])
      femaleoky.append(acc_timey[i])
      femaleokz.append(acc_timez[i])
    elif Ys[i] == 1:
      females1x.append(acc_timex[i])
      females1y.append(acc_timey[i])
      females1z.append(acc_timez[i])
    elif Ys[i] == 2:
      females2x.append(acc_timex[i])
      females2y.append(acc_timey[i])
      females2z.append(acc_timez[i])
    elif Ys[i] == 3:
      females3x.append(acc_timex[i])
      females3y.append(acc_timey[i])
      females3z.append(acc_timez[i])

'''
for i in xrange(0,len(Yg)):
  if Yg[i] == 0:
    if Ys[i] == 0: 
      maleokx.append(vehtime[i])
    elif Ys[i] == 1:
      males1x.append(vehtime[i])
    elif Ys[i] == 2:
      males2x.append(vehtime[i])
    elif Ys[i] == 3:
      males3x.append(vehtime[i])
  elif Yg[i] == 1:
    if Ys[i] == 0: 
      femaleokx.append(vehtime[i])
    elif Ys[i] == 1:
      females1x.append(vehtime[i])
    elif Ys[i] == 2:
      females2x.append(vehtime[i])
    elif Ys[i] == 3:
      females3x.append(vehtime[i])


x= 2
'''
D1 = librosa.amplitude_to_db(np.abs(librosa.stft(maleok[x])), ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(males1[x])), ref=np.max)
D3 = librosa.amplitude_to_db(np.abs(librosa.stft(males2[x])), ref=np.max)
D4 = librosa.amplitude_to_db(np.abs(librosa.stft(males3[x])), ref=np.max)
''' 

plt.subplot(411)
plt.title('Ok google')
plt.plot(maleokx[x],color ='blue',label='x-axis')
#plt.plot(maleoky[x],color ='green',label='y-axis')
#plt.plot(maleokz[x],color ='red',label='z-axis')
#plt.xlabel('time')
plt.xticks(rotation=90)
plt.ylabel('Amplitude')
plt.legend(bbox_to_anchor=(1,1.5))

'''
plt.subplot(412)
plt.title('Fine thank you')
plt.plot(males1x[x],color ='blue',label='x-axis')
plt.plot(males1y[x],color ='green',label='y-axis')
plt.plot(males1z[x],color ='red',label='z-axis')
#plt.xlabel('time')
plt.ylabel('Amplitude')



plt.subplot(413)
plt.title('How are you')
plt.plot(males2x[x],color ='blue',label='x-axis')
plt.plot(males2y[x],color ='green',label='y-axis')
plt.plot(males2z[x],color ='red',label='z-axis')
#plt.xlabel('time')
plt.ylabel('Amplitude')



plt.subplot(414)
plt.title('Good morning')
plt.plot(males3x[x],color ='blue',label='x-axis')
plt.plot(males3y[x],color ='green',label='y-axis')
plt.plot(males3z[x],color ='red',label='z-axis')
#plt.xlabel('time')
plt.ylabel('Amplitude')

plt.subplots_adjust(hspace=.99)
#plt.legend(loc='lower right')

'''

plt.savefig('veh rotated.pdf',dpi=300)
plt.show()


'''
for x in xrange(1,len(X1veh)):
  s1 = X1veh[x]
  y = np.array(data_veh_time[x])
  print(Yg[x],Yh[x],Ys[x])
  hop_length = 44521
  sr =1000
  plt.figure(figsize=(12, 8))
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

  plt.subplot(211)
  plt.title('veh signal')
  plt.plot(y)
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  
  plt.subplot(412)
  plt.title('veh signal')
  plt.specgram(s1,Fs=1000)
  plt.colorbar(format='%+2.0f dB')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')

  plt.subplot(212)
  display.specshow(D,y_axis='log', x_axis='time')
  plt.title('Power spectrogram')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()
  plt.show()

  #filename = './'+ str(Yg[x])+str(Yh[x])+str(Ys[x])+ 'spec.pdf'
  #plt.savefig(filename, dpi=300)

  plt.subplot(411)
  plt.title('veh signal')
  plt.plot(s1)
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')

  plt.subplot(412)
  plt.title('interplation')
  plt.plot(s2)
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')
  
  plt.subplot(413)
  plt.title('frequency domain')
  plt.plot(s1)
  plt.xlabel('Sample')
  plt.ylabel('Amplitude')   

  plt.subplot(414)
  plt.specgram(s1,Fs=500)
  plt.xlabel('Time')
  plt.ylabel('Frequency')
  plt.show()

  plt.legend()
  filename = '../files/plots/'+ parts[i]+ 'feature_importances.pdf'
  fig.savefig(filename, dpi=300)
  '''