
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:55:39 2017

@author: slim
"""
import matplotlib.pyplot as plt
import numpy as np
#Getting the x array
x=np.arange(0,5.1,0.1)
#plotting the function
plt.plot(x,2*(x**2)-20*x+1)
plt.show()


#%%

#taking 1 element over 8 from x array
trainingset=[x[0]]
i=1
while i*8<x.size:
    trainingset.append(x[i*8])
    i+=1
    

mu, sigma=0,4

#We generate all the training examples adding some noise from a normal distribution N(mu,sigma)
target=[]
for i in range(0,len(trainingset)):
    target.append((2*(x[i*8]**2)-20*x[i*8]+1)+np.random.normal(mu,sigma))

#Plot the noisy trainingset and the original function
plt.scatter(trainingset,target)
plt.plot(x,2*(x**2)-20*x+1)
plt.show()
#%%


w=np.ones(d+1)

Error=-1
plt.plot(x,2*(x**2)-20*x+1)

plt.scatter(trainingset,target) 
plt.show()
#%%
d=2
lrate=0.002

#Phi is the matrix with homogeneous coordinates containg the trainingsets elements x^1,...,x^d
phi=np.array([[1,1,1,1,1,1,1],trainingset,np.asarray(trainingset)**2])

#Computing and plotting the closedForm solution depending on the trainingset
clauseform=np.dot(np.dot(np.linalg.inv(np.dot(phi,phi.T)),phi),np.asarray(target).T)
plt.plot(x,2*(x**2)-20*x+1)
plt.scatter(trainingset,target)
plt.plot(x,clauseform[2]*(x**2)+clauseform[1]*x+clauseform[0])
plt.show()

#%%
# Initializing the weight vector W
w=np.zeros(d+1)

#800 Max Iterations
for j in range(0,8): 
    print("============================================================")
    #For each training examples:
    for i in range(0,len(trainingset)):
        #We take the homogenous vector
        phi=np.asarray([1,trainingset[i],trainingset[i]**2])
        #Compute the inner product between the weight vector and the current training example
        res=np.dot(w,phi)
        #Compute the squared difference
        Error=(target[i]-res)**2
        #Updating the weight vecto
        w+=lrate*(target[i]-np.dot(w,phi))*phi
    print(j)
    #printing the new regression curve
    plt.plot(x,2*(x**2)-20*x+1)
    plt.scatter(trainingset,target)
    plt.plot(x,w[2]*(x**2)+w[1]*x+w[0])
    plt.plot(x,clauseform[2]*(x**2)+clauseform[1]*x+clauseform[0])
    plt.show()
    print("============================================================")
                

#%%
# Max Dimension
maxD=9
# Max Iteration
maxIt=2000
# The point we are going to evaluate our models on
x_star=2
#Standard Deviation for the noise injection
sigma=4
# Original value of x_star with the original function
f_x_original=2*(x_star**2)-20*x_star+1

#Initialization of all the evaluation arrays
error_array_MSE=np.zeros(maxD)
error_array_bias=np.zeros(maxD)
error_array_var=np.zeros(maxD)


for i in range(0,maxIt):
    target=[]
    # Generation of the noisy sample
    for i in range(0,len(trainingset)):
        target.append((2*(x[i*8]**2)-20*x[i*8]+1)+np.random.normal(mu,sigma))
        
    phi=np.array([[1,1,1,1,1,1,1]])
    
    # For each dimension
    for d in range(1,maxD): 
        #Generating the homogenous training points 
        phi=np.vstack((phi,np.asarray(trainingset)**(d)))
        #Computing the closed forme
        clauseform=np.dot(np.dot(np.linalg.inv(np.dot(phi,phi.T)),phi),np.asarray(target).T)
        

        #Adding the constant
        f_x_closeF=clauseform[0]
        #Summing all theelement-wise products between phi and w
        for j in range(1,d):
            f_x_closeF+=clauseform[j]*(x_star**j)
        #Computing the evaluation values and summing them for each dimensions
        error_MSE=(f_x_original-f_x_closeF)
        error_array_MSE[d]+=error_MSE
        error_array_bias[d]+=((f_x_closeF)/maxIt)
        error_array_var[d]+=f_x_closeF
        
#Getting the square of the empirical mean of the difference which is MSE
for i in range(0,maxD):
    error_array_MSE[i]=(error_array_MSE[i]/maxIt)**2


# Computting the variance
error_array_var -= error_array_bias
for i in range(0,maxD):
    error_array_var[i]=(error_array_var[i]**2)/maxIt



#And the Bias^2
error_array_bias=f_x_original-error_array_bias
for i in range(0,maxD):
    error_array_bias[i]=error_array_bias[i]**2


#%%
#Ploting the evaluation functions
plt.plot(np.arange(1,maxD,1),error_array_MSE[1:maxD],label="MSE")
plt.plot(np.arange(1,maxD,1),error_array_bias[1:maxD],label="Bias^2")
plt.plot(np.arange(1,maxD,1),error_array_var[1:maxD],label="Variance")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    
    
