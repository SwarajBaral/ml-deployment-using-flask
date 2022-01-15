import pickle
from flask import Flask, request, render_template
import numpy as np

mod = open("iris_model.pickle", "rb")
model = pickle.load(mod)

SPECIES = {
	0: 'Iris-Setosa',
	1: 'Iris-Versicolour',
	2: 'Iris-Virginica',
}

# prediction_data = [[6.1,3.0,4.9,1.8]]
# pred = model.predict(prediction_data)
# print(round(pred[0][0]))

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
	return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
	features = [float(x) for x in request.form.values()]
	final = [np.array(features)]
	ans = model.predict(final)
	ans = ans[0][0]
	if ans < 0:
		ans = 0
	elif ans > 2:
		ans = 2
	else:
		ans = round(ans)
	return render_template("home.html", pred=f"Your species is {SPECIES[ans]}", val=features)
if __name__ == '__main__':
	app.run(debug=True)