# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start the program
2. Initialize Parameters
3. Hypothesis Function (Sigmoid Function)
4. Cost Function (Log-Loss)
5. Gradient Calculation
6. Parameter Update
7. Repeat for Multiple Iterations
8. Prediction Function
9. Model Evaluation
10. End the program

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SARON XAVIER A
RegisterNumber:  212223230197
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Placement_Data.csv")
dataset

# dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

# categorising col, for further labeling
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=da
taset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset

#deleting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y

#initialize the model parameters.
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1 /(1+np.exp(-z))

#define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

# define the gradient descent algorithm
def gradient_descent(theta, X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
#train the model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#Make predictions.
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)

#evaluate the model.
accuracy=np.mean(y_pred.flatten()==y)
print('Acurracy:',accuracy)

print(y_pred)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

## Accuracy:
![exp5out](https://github.com/user-attachments/assets/b7ae5aaf-818e-4a29-a22a-fd8cb0e07414)

## Y-pred:
![exp5out1](https://github.com/user-attachments/assets/dc3f23ff-3a06-4c39-9c38-468578972216)

## New Y-pred:
![exp5out3](https://github.com/user-attachments/assets/ea128c84-a5f3-47f4-81fc-82e427b0c294)

## New Y-Pred:
![exp5out3](https://github.com/user-attachments/assets/7e23d6a2-ebe5-4bab-b879-d15921c4df22)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

