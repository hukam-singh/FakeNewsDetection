from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    if news.strip() == "":
        return render_template('result.html', prediction="Please enter news text.")

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)[0]

    result = "Real News 🟢" if prediction == 1 else "Fake News 🔴"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
