from flask import Flask, render_template, request
from sklearn.decomposition import PCA
from joblib import load
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index1.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        pca = load('pca.joblib')
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = float(request.form['rbc'])
        pc = float(request.form['pc'])
        pcc = float(request.form['pcc'])
        ba = float(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wc = float(request.form['wc'])
        rc = float(request.form['rc'])
        htn = float(request.form['htn'])
        dm = float(request.form['dm'])
        cad = float(request.form['cad'])
        appet = float(request.form['appet'])
        pe = float(request.form['pe'])
        ane = float(request.form['ane'])
        values = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])
        values_pca = pca.transform(values)
        prediction = model.predict(values_pca)
        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
