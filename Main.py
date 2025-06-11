from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Depression Detection using EEG") #designing main screen
main.geometry("1000x650")

global dataset, X, Y, classifier
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, cnn

def loadDataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    dataset.Label[dataset.Label == 2.0] = 1.0 
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.title("Total Normal & Depressed Records found in dataset 0 (Normal) & 1 (Depressed)")
    plt.show()

def featuresExtraction():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    text.insert(END,"Extracted Features from EEG Signals\n\n")
    text.insert(END, str(X)+"\n\n")
    text.insert(END,"Total features found in each records : "+str(X.shape[1])+"\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details. 80% dataset used for training & 20% dataset used for testing\n\n")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records used to train algorithms are : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test algorithms are  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    conf_matrix = confusion_matrix(testY, predict) 
    text.insert(END,algorithm+' Accuracy    : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FScore      : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ['Normal', 'Depressed']
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    text.delete('1.0', END)
    svm_cls = SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test) 
    calculateMetrics("SVM", predict, y_test)
    
def runCNN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y, cnn
    global accuracy, precision, recall, fscore
    X = X[:,0:972]
    XX = X.reshape(X.shape[0], 18, 18, 3)
    YY = to_categorical(Y)
    print(XX.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(XX, YY, test_size=0.2)

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()
        cnn.load_weights("model/model_weights.h5")
        cnn._make_predict_function()   
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 3, 3, input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))
        cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = y_test1.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(X_train1, y_train1, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test1, y_test1))
        cnn.save_weights('model/model_weights.h5')            
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn.summary())
    predict = cnn.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test1, axis=1)
    calculateMetrics("CNN", predict, testY)
    
def predictDepression():
    global cnn
    labels = ["Normal", "Depressed"]
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    dataset = dataset.values
    testData = dataset[:,0:972]
    test_X = testData.reshape(testData.shape[0], 18, 18, 3)
    predict = cnn.predict(test_X)
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(testData)):
        text.insert(END,str(testData[i])+" PREDICTED AS ====> "+labels[predict[i]]+"\n\n")

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['CNN','Precision',precision[1]],['CNN','Recall',recall[1]],['CNN','F1 Score',fscore[1]],['CNN','Accuracy',accuracy[1]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("Existing SVM & Propose CNN Performance Graph")
    plt.show()

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Depression Detection using EEG', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload EEG-Signal Dataset", command=loadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

featuresButton = Button(main, text="Features Extraction", command=featuresExtraction)
featuresButton.place(x=10,y=150)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Run Existing SVM Algorithm", command=runSVM)
svmButton.place(x=10,y=200)
svmButton.config(font=font1) 

cnnButton = Button(main, text="Run Propose CNN Algorithm", command=runCNN)
cnnButton.place(x=10,y=250)
cnnButton.config(font=font1)

predictButton = Button(main, text="Predict Depression from Test Signals", command=predictDepression)
predictButton.place(x=10,y=300)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=10,y=350)
graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=10,y=400)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=115)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=320,y=100)
text.config(font=font1)

main.config(bg='light blue')
main.mainloop()
