import time, os, sys
import struct
from os import listdir
from os.path import isfile, join
import ctypes, math
import csv, struct
import numpy, scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz,lp2hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import librosa
import librosa.display



def process_vect(y,n): 
  #w = savgol_filter(data[k], 101, 2)
  b = [1.0 / n] * n
  a = 1
  yy = lfilter(b,a,y)
  return yy

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


def get_array_magnitude(arr1,arr2,arr3):
  #arr1,_ = librosa.effects.trim(arr1)
  arr1 /= numpy.max(numpy.abs(arr1),axis=0)
  arr2 /= numpy.max(numpy.abs(arr2),axis=0)
  arr3 /= numpy.max(numpy.abs(arr3),axis=0)
  for i in xrange(0,3):
    mag = []
    for j in range(0,len(arr1)):
            temp = []
            temp.append(arr1[j])
            temp.append(arr2[j])
            temp.append(arr3[j])
            m = numpy.linalg.norm(temp)
            mag.append(m)
    #plt.plot(arr1,color='red')
    #plt.plot(arr2,color='blue')
    #plt.plot(arr3,color='green')
    #plt.plot(mag)
    #plt.show()
    return mag

# read veh data and accelerometer data

def file_array(file,type): # read a file and write to array (type = 'veh' or 'accelerometer')
  array=[]
  list=[]
  with open(file, 'r') as f: #open file
    csv_reader = csv.reader(f, delimiter=',')

    if type == 'veh':
      array=numpy.loadtxt(file,dtype=str,delimiter=',',skiprows=0,usecols=(0,))
      #plt.xticks(rotation=90)

      #plt.plot(arr2,color='blue')
      #plt.plot(arr3,color='green')
      #plt.plot(mag)
    elif type == 'accelerometer':
      array1=numpy.loadtxt(file,dtype=str,delimiter=',',skiprows=0,usecols=(1,))
      array2=numpy.loadtxt(file,dtype=str,delimiter=',',skiprows=0,usecols=(2,))
      array3=numpy.loadtxt(file,dtype=str,delimiter=',',skiprows=0,usecols=(3,))
      array1 = map(float,array1)
      array2 = map(float,array2)
      array3 = map(float,array3)

      # compute the vector average and append to array
      array = get_array_magnitude(array1, array2, array3)
      #array.append(arr)
      #array.append(array1)
      #array.append(array2)
      #array.append(array3)
  return array


def path_to_list(pathF):
  list=[]
  listarray=[]
  for filename in os.listdir(pathF): #filename             
    pathN=pathF+filename # absolute path for every file
    list = file_array(pathN) # return VEH vector
    listarray.append(list)
    return listarray



# get labels
'''
def get_gender(name):
  if name[0] == 'M':
    label=0
  elif name[0] == 'F':
    label=1  
  return label

def get_hot_ornot(name):

  if name is "OkayGoogle":
    label=0
  else: 
    label=1
  return label


def get_sentence(name):

  if name is "OkayGoogle": 
    label=0
  elif name is "FineThankYou": 
    label=1
  elif name is "HowAreYou": 
    label=2
  elif name is "GoodMorning": 
    label=3 
  return label

  #return label
'''
def read_data(path,mod):
  
  file=path + "file.txt"
  list1 = ["flatTop","verticalTop"]
  list2 = ["FineThankYou","GoodMorning","HowAreYou","OkayGoogle"]
  with open(file) as f:

     list0 = f.readlines()
     list0 = [x.strip() for x in list0] # individuals
     listdataveh=[]
     listdataacc=[]
     listdataaccx=[]
     listdataaccy=[]
     listdataaccz=[]

     Y1=[]
     Y2=[]
     Y3=[]
     for i in range(0, len(list0)):
      path2=path+list0[i]+'/'
      #print "path2=",path2
      for j in range(0, len(list1)): #for the both directions
         path3=path2+list1[j]+"/" 
         #print "path3=",path3
         for k in range(0, len(list2)): #for all sentences
          path4=path3+list2[k]+"/"
          #print "path4=",path4
          for filename in os.listdir(path4): #filename
            list3=[] 
            path5=path4+filename # absolute path for every file
            list3veh = file_array(path5,'veh') # return VEH or accelerometer vector (temporal domain)
            list3veh=map(int,list3veh)
            listsacc = file_array(path5,'accelerometer') # return VEH or accelerometer vector (temporal domain)
            #list3acc=map(int,listsacc)
            #accx=map(int,listsacc[1])
            #accy=map(int,listsacc[2])
            #accz=map(int,listsacc[3])
              
            listdataveh.append(list3veh)
            listdataacc.append(listsacc)
            #listdataaccx.append(accx)
            #listdataaccy.append(accy)
            #listdataaccz.append(accz)

            if list0[i][0] == 'M':
              label1=0
              #print("----------------------------------",list0[i][0])
            elif list0[i][0] == 'F':
              label1=1  
            if list2[k] is "OkayGoogle":
              label2=0
              label3=0
            elif list2[k] is "FineThankYou": 
              label2=1
              label3=1
            elif list2[k] is "HowAreYou": 
              label2=1
              label3=2
            elif list2[k] is "GoodMorning": 
              label2=1
              label3=3 

            #label1=get_gender(list2[k])
            #label2=get_hot_ornot(list2[k])
            #label3=get_sentence(list0[i])
            Y1.append(label1)
            Y2.append(label2)
            Y3.append(label3)
            #print "user :",list0[i]," dir :",list1[j]," sentence: ",list2[k], "label: ", label1 , label2, label3
  
  #return listdataveh,listdataacc,listdataaccx,listdataaccy,listdataaccz,Y1,Y2,Y3
  return listdataveh,listdataacc,Y1,Y2,Y3


def file_arrayg(file): # read a file and write to array (type = 'veh' or 'accelerometer')
  array=[]
  with open(file, 'r') as f: #open file
    csv_reader = csv.reader(f, delimiter=',')
    array1=numpy.loadtxt(file,dtype=str,delimiter=' ',skiprows=0,usecols=(1,))
    array2=numpy.loadtxt(file,dtype=str,delimiter=' ',skiprows=0,usecols=(2,))
    array3=numpy.loadtxt(file,dtype=str,delimiter=' ',skiprows=0,usecols=(3,))
    array1 = map(float,array1)
    array2 = map(float,array2)
    array3 = map(float,array3)

    # compute the vector average and append to array
    arr = get_array_magnitude(array1, array2, array3)
  return arr

def labelsg(n):
  if n=='0':
    l = 0
  if n=='1':
    l = 1
  if n=='2':
    l = 2
  if n=='3':
    l = 3
  if n=='4':
    l = 4
  if n=='5':
    l = 5
  if n=='6':
    l = 6
  if n=='7':
    l = 7
  if n=='8':
    l = 8
  if n=='9':
    l = 9
  if n=='O':
    l = 10
  if n=='Z':
    l = 11
  return l



digits = []
gender = []

data_gyro = []
def read_grodata(path):  
    data_gyro_x=[]
    data_gyro_y=[]
    data_gyro_z=[]
    for filename in os.listdir(path): #filename
        file = path+filename
        s1 = filename[0]
        s = filename[4]
        y1 = labelsg(s)
        if s1 == 'F':
          y2=0
        if s1 =='M':
          y2=1
        digits.append(y1)
        gender.append(y2)
        gyr = file_arrayg(file)
        #list32=map(int,list32)             
        data_gyro.append(gyr)
        #data_gyro_y.append(gyry)
        #data_gyro_z.append(gyrz)
    return data_gyro,digits,gender
