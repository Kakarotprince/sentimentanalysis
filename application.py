from flask import Flask,request, render_template
import pickle
import nltk
nltk.download('wordnet')
app = Flask(__name__, template_folder='templates') 
model=pickle.load(open('./LogReg.pickle','rb')) 
vectoriser = pickle.load(open('./vectoriser.pickle','rb')) 
import dill  
preprocess = dill.load(open('./PreProcess.dill','rb')) 
 
@app.route('/') 
def hello_world():
    return render_template("index.html",word='{Enter text}',pred='{Enter text}')
 
 
@app.route('/predictLR', methods=['POST', 'GET'])
def predict():
    i = ''
    for key, value in request.form.items():
        if key == 'text':
            i = value
    text = [i]
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict_proba(textdata)
    pp = sentiment[0][1]

    if pp > 0.90:
        return render_template('index.html', word=i, pred='Super Positive')
    elif pp > 0.60:
        return render_template('index.html', word=i, pred='Positive')
    elif pp > 0.40:
        return render_template('index.html', word=i, pred='Neutral')
    elif pp > 0.10:
        return render_template('index.html', word=i, pred='Negative')
    else:
        return render_template('index.html', word=i, pred='Super Negative')

 
if __name__ == '__main__': 
    app.run(debug=False) 
