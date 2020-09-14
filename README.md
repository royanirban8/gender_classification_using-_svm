# gender_classification_using-_svm
This project classifies gender i.e, male or female using support vector machines. The Faces94 dataset is being used in this project.


In this project the main problem statement is to perform efficient gender classification on low resolution images and images with minimal information of human, for achieving minimal information on our dataset we preprocess data so that the information about hair is removed and all images are preprocessed so that they don’t have hairs and also their resolution is minimized. When we have such minimal information about the subject then our system has to be strong enough to classify the subject under consideration into proper gender so that error is minimized, also it is seen that difference in error rate of classifying a low resolution image and high resolution image is found to be less in our system when we compare it with human test subjects. Also humans have difficulties in classifying images with no hairs and in such case error is high so we design such system to classify images with no hair and low resolution with minimum error. In general in gender classifier we have an input facial image x which generates a scalar output f(x) whose polarity sign determines the class of subject Here we will be using support vector machines(SVM) for classification of gender and compare its results with other classifiers such as linear, quadratic, fisher linear discriminant, nearest neighbor, radial basis functions(RBF) networks and large ensemble RBF classifier. SVM is an algorithm for pattern classification and regression and here we mainly find an optimal linear hyperplane which leads to proper classification and error of classification is minimized. In case of linearly non separable data SVM map the input to high dimensional feature space where a linear hyper plane can be found.

Here we have used feret database in the paper which had images with resolution 256 by 384 pixel, while performing pre-processing we compensate for translation, scaling and rotation and then after full pre-processing we have our final image with minimal or no hair information and final resolution will be 21 by 12 pixel for our low resolution experiments and then after splitting the data into training and test data we apply SVM with different kernel and it is found through repeated experiments that we get minimal error when we use Gaussian RBF kernel followed by cubic polynomial kernel.

ALGORITHM:

1. We import the images of males and females from our dataset of faces94 and preprocessing which includes reshape and conversion to grayscale.

2. We scale all the images so that further processing on them becomes easier as they are all scaled to same levels and then principal component analysis(PCA) is performed on them to get the four principal components of the images.

3. Now we create a data frame which has labels for every image as in above steps only labeling was done for male and female images.

4. Now we split data in training and test data set and then we apply Support vector machine(SVM) with different kernel and we measure the error with all kernel and then their final comparison is done.

Code:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage import io,color
%matplotlib inline

numImg = 20
numSbj = 19
A = np.zeros([2 * numImg * numSbj,180 * 200])
y = np.zeros([2 * numImg * numSbj])
c = 0

fPath = 'male'
j = numSbj
for i in os.listdir(fPath):
    if(j <= 0):
        break
    j = j - 1
    for f in os.listdir(fPath + '/' + i):
        imgPath = fPath + '/' + i + '/' + f
        A[c, :] = color.rgb2gray(io.imread(imgPath)).reshape([1,180 * 200])
        y[c] = 0
        c = c + 1

fPath = 'female'
j = numSbj
for i in os.listdir(fPath):
    if(j <= 0):
        break
    j = j - 1
    for f in os.listdir(fPath + '/' + i):
        imgPath = fPath + '/' + i + '/' + f
        A[c, :] = color.rgb2gray(io.imread(imgPath)).reshape([1,180 * 200])
        y[c] = 1
        c = c + 1
 
 from sklearn.preprocessing import StandardScaler
 
 scaler = StandardScaler()
 
 scaler.fit(A)
 
 A = scaler.transform(A)
 
 from sklearn.decomposition import PCA
 
 p = PCA(n_components=4,random_state=101)
 
 p.fit(A)
 
 A = p.transform(A)
 
df1 = pd.DataFrame(A)
d = {'o/p':y}
df2 = pd.DataFrame(d)
df = df1.join(df2)
df.head()

from sklearn.model_selection import train_test_split

X = df.drop('o/p',axis=1)
y = df['o/p']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

SVM

(i) kernel = rbf

from sklearn.svm import SVC

s_rbf = SVC(C=1.0,kernel='rbf',gamma='scale',random_state=101)

s_rbf.fit(X_train,y_train)

y_pred_rbf = s_rbf.predict(X_test)

(ii)kernel=cubic polynomial

s_poly = SVC(C=1.0,kernel='poly',degree=3,gamma='scale',random_state=101)

s_poly.fit(X_train,y_train)

y_pred_poly = s_poly.predict(X_test)

(iii)kernel=linear

s_linear = SVC(C=1.0,kernel='linear',degree=3,gamma='scale',random_state=101)

s_linear.fit(X_train,y_train)

y_pred_linear = s_linear.predict(X_test)

Comparison:

from sklearn.metrics import accuracy_score

print('accuracy of svm with gaussian rbf kernel =',accuracy_score(y_pred_rbf,y_test))
print('accuracy of svm with cubic poly kernel =',accuracy_score(y_pred_poly,y_test))
print('accuracy of svm with linear kernel =',accuracy_score(y_pred_linear,y_test))




