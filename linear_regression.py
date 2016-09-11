# coding: utf-8
# author: thariq khalid
# X' denotes X transpose

import numpy as np
#setting the data
x = np.linspace(0,10,100)
y = np.sin(x) + np.random.normal(0,0.05,100)

#plot the data
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()

#creating the features matrix of the polynomial form [1,x,x^2]
X = np.ones((100,3))
X[:,1]= x
X[:,2] = x**2

# These are the ideal W coefficients acheived by minimizing the error; i.e dE/dW = 0
# W = inv(X'X)X'Y
global_W = np.dot(np.dot(np.linalg.inv(np.dot(np.matrix.transpose(X),X)),np.matrix.transpose(X)),y)

alpha = 0.00000025 #learning rate
iterations = 1500
W = np.random.rand(3,) #creating a random co-efficients vector

#gradient descent
#This is of the form W(t+1) = W(t) - alpha*dE/dW
#which simplifies to W(t+1) = W(t) - alpha*(-X'Y + X'XW) using first principles
# The 3 lines of code
for i in range(iterations):
    finalW = W - alpha*(-np.dot(np.matrix.transpose(X),y) + np.dot(np.dot(np.matrix.transpose(X),X),W))
    W = finalW

print finalW,global_W




