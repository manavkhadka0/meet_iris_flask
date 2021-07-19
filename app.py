from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def pred():
    data1 = request.form['SL']
    data2 = request.form['SW']
    data3 = request.form['PL']
    data4 = request.form['PW']
    arr = np.array([[data1,data2,data3,data4]])
    output = model.predict(arr)
    return render_template('output.html',data=output)

if __name__ == '__main__':
    app.run(debug=True)
