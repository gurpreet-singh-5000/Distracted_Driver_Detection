pip install scikit-image


import numpy as np 
import pandas as pd 
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

dataset = pd.read_csv('./driver_imgs_list.csv')
dataset.head(5)

dataset['img']

img=cv2.imread("imgs/train/"+str(dataset['classname'][0])+"/"+str(dataset['img'][0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
img=cv2.resize(img,(320,240))

img=cv2.imread("imgs/train/"+str(dataset['classname'][0])+"/"+str(dataset['img'][0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
img=cv2.resize(img,(64,64))

plt.imshow(img)

ppc=2
cpb=1
fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),block_norm= 'L2',visualize=True)
plt.imshow(hog_image)

ppc=4
cpb=2
fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),block_norm= 'L2',visualize=True)
plt.imshow(hog_image)

ppc=16
cpb=4
fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),block_norm= 'L2',visualize=True)
plt.imshow(hog_image)

fd


hog_image

target_names=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']

train_images=[]
train_labels=[]
for i in range(0,22424):
    imge=cv2.imread("imgs/train/"+str(dataset['classname'][i])+"/"+str(dataset['img'][i]))
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    imge=cv2.resize(imge,(64,64))
    train_images.append(imge)
    train_labels.append(dataset['classname'][i])

len(train_images)
train_images[0].shape
len(train_labels)

ppc=2
cpb=1
hog_features=[]
hog_images=[]
for i in range(0,22424):
    fd,hog_image = hog(train_images[i], orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),block_norm= 'L2',visualize=True)
    hog_features.append(fd)
    hog_images.append(hog_image)

plt.imshow(hog_images[20000])


hog_features=np.array(hog_features)

hog_features.shape

df=pd.DataFrame(hog_features)

df['labels']=train_labels

df.shape

#Truncated the dataset from 22424 images to 4000 images

df_trunc=df.sample(frac=1).reset_index(drop=True)[:4000]

df_trunc.shape

train_labels_trunc=df_trunc['labels']
df_trunc=df_trunc.drop(columns='labels')

from sklearn.decomposition import PCA
n_comp=[500,1000,1500,2000]
exp_var=[]

for j in n_comp:
    pcaApp=PCA(n_components=j)
    df_trunc_pca=pcaApp.fit_transform(df_trunc)
    listp=pcaApp.explained_variance_ratio_
    sumP=0
    for k in listp:
        sumP+=k
    exp_var.append(sumP)
    print(sumP)

plt.scatter(n_comp,exp_var)
plt.xlabel('Number of components')
plt.ylabel('Variance Explained')

pcaApp=PCA(n_components=2000)
df_trunc_pca=pcaApp.fit_transform(df_trunc)
listp=pcaApp.explained_variance_ratio_
sumP=0
for k in listp:
    sumP+=k
print(sumP)

df_trunc_pca.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df_trunc_pca, train_labels_trunc, test_size=0.3)

from sklearn.svm import SVC

clf = svm.SVC(kernel='rbf',C=1)
clf.fit(x_train,y_train);

print(classification_report(y_test, clf.predict(x_test), target_names=target_names))

clf = svm.SVC(kernel='rbf',C=10)
clf.fit(x_train,y_train);
print(classification_report(y_test, clf.predict(x_test), target_names=target_names))

clf = svm.SVC(kernel='linear',C=1)
clf.fit(x_train,y_train);
print(classification_report(y_test, clf.predict(x_test), target_names=target_names))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
print(classification_report(y_test, clf.predict(x_test), target_names=target_names))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(x_train,y_train)
print(classification_report(y_test, clf.predict(x_test), target_names=target_names))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10).fit(x_train,y_train)
print(classification_report(y_test, neigh.predict(x_test), target_names=target_names))

