import os
import numpy as np
import pickle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import svm
from lightgbm import LGBMClassifier
from django.shortcuts import render
from django.contrib import messages

# Load datasets and models
X = np.load("model/X.txt.npy")
Y = np.load("model/Y.txt.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

with open('model/tfidf.txt', 'rb') as file:
    tfidf = pickle.load(file)

X = tfidf.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Load or train SVM model
if os.path.exists('model/svm.txt'):
    with open('model/svm.txt', 'rb') as file:
        svm_cls = pickle.load(file)
else:
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)

# Load or train LightGBM model
if os.path.exists('model/lgbm.txt'):
    with open('model/lgbm.txt', 'rb') as file:
        lgbm_cls = pickle.load(file)
else:
    lgbm_cls = LGBMClassifier()
    lgbm_cls.fit(X_train, y_train)
    with open('model/lgbm.txt', 'wb') as file:
        pickle.dump(lgbm_cls, file)

# Load RandomForest model
with open('model/rf.txt', 'rb') as file:
    rf_cls = pickle.load(file)

def RunSVM(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        precision = []
        accuracy = []
        fscore = []
        recall = []

        # SVM model prediction
        predict = svm_cls.predict(X_test)
        acc = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)

        # Confusion matrix plot for SVM
        conf_matrix = confusion_matrix(y_test, predict)
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(conf_matrix, xticklabels=['Normal URL', 'Phishing URL'], yticklabels=['Normal URL', 'Phishing URL'], annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, 2])
        plt.title("SVM Confusion Matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        model_results = [{
            'model': 'SVM',
            'accuracy': accuracy[0],
            'precision': precision[0],
            'recall': recall[0],
            'fscore': fscore[0]
        }]

        context = {
            'model_results': model_results,
            'confusion_matrix_img': img_str
        }

        return render(request, 'ViewOutput.html', context)

def RunLGBM(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        precision = []
        accuracy = []
        fscore = []
        recall = []

        # LightGBM model prediction
        predict = lgbm_cls.predict(X_test)
        acc = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)

        # Confusion matrix plot for LGBM
        conf_matrix = confusion_matrix(y_test, predict)
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(conf_matrix, xticklabels=['Normal URL', 'Phishing URL'], yticklabels=['Normal URL', 'Phishing URL'], annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, 2])
        plt.title("LightGBM Confusion Matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        model_results = [{
            'model': 'Light GBM',
            'accuracy': accuracy[0],
            'precision': precision[0],
            'recall': recall[0],
            'fscore': fscore[0]
        }]

        context = {
            'model_results': model_results,
            'confusion_matrix_img': img_str
        }

        return render(request, 'ViewOutput.html', context)

def getData(arr):
    data = ""
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 0:
            data += arr[i] + " "
    return data.strip()

def PredictAction(request):
    if request.method == 'POST':
        global rf_cls, tfidf
        url_input = request.POST.get('t1', False)
        test = []
        arr = url_input.split("/")
        if len(arr) > 0:
            data = getData(arr)
            test.append(data)
            test = tfidf.transform(test).toarray()
            predict = rf_cls.predict(test)
            predict = predict[0]
            output = ""
            if predict == 0:
                output = url_input + " Given URL Predicted as Genuine"
            if predict == 1:
                output = url_input + " PHISHING Detected in Given URL"
            context = {'data': output}
            return render(request, 'Predict.html', context)
        else:
            context = {'data': "Entered URL is not valid"}
            return render(request, 'Predict.html', context)

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context = {'data': 'Welcome ' + user}
            return render(request, 'AdminScreen.html', context)
        else:
            context = {'data': 'Invalid Login'}
            return render(request, 'AdminLogin.html', context)
