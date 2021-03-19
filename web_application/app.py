from flask import Flask,render_template,request
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('salary.pkl','rb'))

label = pickle.load(open('label.pkl','rb'))




@app.route('/')
def man():
    return render_template('index.html')

@app.route('/prediction')
def predict():
    return render_template('prediction.html')



@app.route('/predict', methods=['POST'])
def quality():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
   
    

    items = [ data2, data3, data5, data6, data7, data8, data9, data13 ]
    values = []
    for i in items:
        values.append(i)

    values = label.fit_transform(values)
    items1 = [ data1, data4,data10, data11, data12]
    items2=np.array(items1)
    items3=np.concatenate((items2,values))
    pred = np.array(items3).reshape(1,-1)
    
    predict = model.predict(pred)
    output = predict.item()
    return render_template('result.html',data = output)

if __name__ == "__main__":
    app.run()
