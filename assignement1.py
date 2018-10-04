#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:15:35 2017

@author: slim
"""

        
#%% DATA IMPORTATION

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.options.display.float_format = '{:10,.17f}'.format

df = pd.read_csv('perceptrondata.csv',sep='\s+',header=None)

#plt.scatter(df[0],df[1],c=df[2]) 

df_1=df.as_matrix()
homogeneous=np.ones((200,1))
dataset=np.hstack((homogeneous,df_1[:,0:2]**2))

df_1[0,2]

x=np.arange(-0.1,0.25,0.001)
#%% ONLINE PERCEPTRON PART

    
def percTrain(X, t, maxIt, online):
    w=[0,0,0]
    lrate=0.1
    error = -1
    it=0
    if online == True:            
        while error !=0 and it < maxIt :
            error = 0
            for i in range(0,X[:,1].size-1) :
                res=np.dot(X[i,:],t[i])
                if np.dot(np.array(w).T,res)<=0:
                    w+=res*lrate
                    error += 1
            it+=1
            plt.scatter(df[0]**2,df[1]**2,c=df[2])
            plt.plot(x,-w[0]/w[2]-(w[1]/w[2])*x)
        plt.show()
        print('ERROR : ',error,' it :',it ) 
    else:
        while error !=0 and it < maxIt :
            error = 0
            dw=0
            for i in range(0,X[:,1].size-1) :
                res=np.dot(X[i,:],t[i])
                if np.dot(np.array(w).T,res)<=0:
                    dw=dw+res
                    error += 1
            it+=1
            w+=lrate*dw
#            plt.scatter(df[0]**2,df[1]**2,c=df[2])
#           plt.plot(x,-w[0]/w[2]-(w[1]/w[2])*x)
#       plt.show()    
        print('ERROR : ',error,' it :',it )    
    return w


unit_step = lambda x: -1 if x < 0 else 1


def perc(w, X):
    w=percTrain(dataset[1:201,:],df_1[1:201,2],1000,True)
    return w

w=[0,0,0]
    
y=perc(w,dataset[0,:])



#%%
plt.scatter(df[0],df[1],c=df[2])

#plt.scatter(df[0]**2,df[1]**2,c=df[2])
plt.plot(x,-y[0]/y[2]-(y[1]/y[2])*x)
plt.show()


#%%

