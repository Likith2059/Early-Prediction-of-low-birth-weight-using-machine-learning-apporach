import os
import xgboost as xgb
import pandas as pd
from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['upload folder']='uploads'

@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':
        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])
    global df
    df = pd.read_csv(path)
    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    if request.method == 'POST':
        global scores1,scores2,scores3,scores4
        global df
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        df = pd.read_csv(path)
        global testsize
        # print('hdf')
        testsize =int(request.form['testing'])
        print(testsize)
        # print('hdf')
        global x_train,x_test,y_train,y_test
        testsize = testsize/100
        # print('hdf')
        print(df)
        X = df.drop(['result'],axis = 1)
        y = df.result
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=testsize,random_state=0)
        # print('ddddddcf')
        model = int(request.form['selected'])
        if model == 1:
            dtc = DecisionTreeClassifier()
            model1 = dtc.fit(x_train,y_train)
            pred1 = model1.predict(x_test)
            # print('sdsj')
            scores1 = accuracy_score(y_test,pred1)
            # print('dsuf')
            return render_template('model.html',score = round(scores1,4),msg = 'accuracy',selected  = 'DECISION TREE CLASSIFIER')
        elif model == 2:
            rfc = RandomForestClassifier()
            model2 = rfc.fit(x_train,y_train)
            pred2 = model2.predict(x_test)
            scores2 =accuracy_score(y_test,pred2)
            return render_template('model.html',msg = 'accuracy',score = round(scores2,3),selected = 'RANDOM FOREST CLASSIFIER')
        elif model == 3:
            svc = SVC()
            model3 = svc.fit(x_train,y_train)
            pred3 = model3.predict(x_test)
            scores3 = accuracy_score(y_test,pred3)
            return render_template('model.html',msg = 'accuracy',score = round(scores3,3),selected = 'SUPPORT  VECTOR  CLASSIFIER ')
        elif model == 4:
            xgbc = xgb.XGBClassifier()
            model4 = xgbc.fit(x_train,y_train)
            pred4 = model4.predict(x_test)
            scores4 = accuracy_score(y_test,pred4)
            return render_template('model.html',msg = 'accuracy',score = round(scores4,3),selected = 'XGBOOST CLASSIFIER')

    return render_template('model.html')


@app.route('/prediction',methods = ['POST',"GET"])
def prediction():
    if request.method == 'POST':
        a =float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = int(request.form['i'])
        # print('ads')
        values = [[float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i)]]
        # print('sjbd')
        dtc = DecisionTreeClassifier()
        model = dtc.fit(x_train,y_train)
        # print('ddfg')
        pred = model.predict(values)
        # print('asdfg')
        return render_template('prediction.html',msg ='success',result = pred)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)