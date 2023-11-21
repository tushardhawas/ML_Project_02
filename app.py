from flask import Flask, request, url_for, redirect, render_template
import warnings
import pickle
warnings.filterwarnings("ignore")
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("house_price.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        # print(int_features)
        # print(final)
        prediction = model.predict(final)

        return render_template('house_price.html', pred=f'House price according to ML model is {prediction} /- Rs.')
    else:
        return render_template('house_price.html')    


if __name__ == '__main__':
    app.run(debug=True)

