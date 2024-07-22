
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import re
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import cv2

from sklearn.decomposition import PCA

main = tkinter.Tk()
main.title("Agricultural Land Image Classification using Regression Neural Network and Compare with KNN for Feature Extraction ")
main.geometry("1300x1200")

global filename
global regression_acc,knn_acc
global classifier
global X,Y
global X_train, X_test, y_train, y_test
global pca
labels = ['Urban Land','Agricultural Land','Range Land','Forest Land']

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="model")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def extractFeatures():
    global X,Y
    global X_train, X_test, y_train, y_test
    global pca
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    
    X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
    text.insert(END,"Number of features in images before applying PCA feature extraction : "+str(X.shape[1])+"\n")
    pca = PCA(n_components = 100)
    X = pca.fit_transform(X)
    text.insert(END,"Number of features in images after applying PCA feature extraction : "+str(X.shape[1])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total Images Found in dataset : "+str(len(X))+"\n")
    

def runKNN():
    text.delete('1.0', END)
    global knn_acc
    global classifier
    global X_train, X_test, y_train, y_test
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    knn_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"KNN Accuracy After applying PCA Feature Extraction : "+str(knn_acc)+"\n")
    classifier = cls

def runRegression():
    global regression_acc
    global X_train, X_test, y_train, y_test
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    regression_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Regression Neural Network Accuracy After applying PCA Feature Extraction : "+str(regression_acc)+"\n")
    
    
def graph():
    height = [regression_acc,knn_acc]
    bars = ('Regression Accuracy', 'KNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    filename = filedialog.askopenfilename(initialdir="sampleImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    img = np.reshape(img, (img.shape[0],(img.shape[1]*img.shape[2]*img.shape[3])))
    img = pca.transform(img)
    predict = classifier.predict(img)
    predict = predict[0]
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Land Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Land Classified as : '+labels[predict], img)
    cv2.waitKey(0)
    
font = ('times', 14, 'bold')
title = Label(main, text='Agricultural Land Image Classification using Regression Neural Network and Compare with KNN for Feature Extraction')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Land Satellite Images", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

featuresButton = Button(main, text="Extract Features from Images", command=extractFeatures)
featuresButton.place(x=50,y=150)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Train & Validate KNN Algorithm", command=runKNN)
svmButton.place(x=310,y=150)
svmButton.config(font=font1) 

nn = Button(main, text="Train & Validate Regression Neural Network", command=runRegression)
nn.place(x=650,y=150)
nn.config(font=font1) 

graphbutton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

predictb = Button(main, text="Upload Test Image & Clasify Lands", command=predict)
predictb.place(x=310,y=200)
predictb.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
