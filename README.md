# 🌟 Sentiment Analysis Web App

A Flask-based web application that performs sentiment analysis on social media text using a Logistic Regression model trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). This app can classify text sentiment ranging from **Super Negative** to **Super Positive** and is fully deployable on **Azure App Service**.



---

## 📁 Project Structure

```
├── flask/
│   ├── LogReg.pickle           # Trained Logistic Regression model
│   ├── vectoriser.pickle       # TF-IDF vectorizer
│   ├── PreProcess.dill         # Preprocessing function
├── templates/
│   └── index.html              # Frontend HTML template
├── app.py                      # Flask application code
├── SentimentAnalysisSocialMedia.py  # Training pipeline
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
```

---

## 🧠 Model Training Pipeline

- **Dataset**: Sentiment140 — 1.6 million tweets labeled as positive (1) or negative (0).
- **Text Preprocessing**:
  - Lowercasing
  - URL and user tag replacement
  - Emoji to text translation
  - Regex cleaning
  - Lemmatization
  - Stopword removal
- **Feature Extraction**: TF-IDF with unigrams and bigrams (max 500k features)
- **Model**: Logistic Regression (`C=2`, `max_iter=1000`)

---

## 🖥️ Web App Functionality

- Input a sentence or tweet.
- The model outputs one of the following:
  - **Super Positive**
  - **Positive**
  - **Neutral**
  - **Negative**
  - **Super Negative**

---

## ⚙️ Installation & Run Locally

### 🔧 Prerequisites

- Python 3.9+
- pip

### 📦 Setup

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧪 Requirements

```
Flask==3.0.3
Werkzeug==3.0.2
dill==0.3.8
scikit-learn==1.4.2
nltk==3.8.1
```

---

## ☁️ Azure Deployment

1. Zip the Flask app and its dependencies.
2. Deploy using Azure App Service (Linux container or manual upload).
3. Ensure all model and vectorizer pickle files are in the root or specified path.

---

## 📊 Results & Evaluation

The Logistic Regression model achieves good performance in classifying tweets with a balanced confusion matrix and high F1-scores across both classes.

Confusion matrix and classification report are visualized during training inside `SentimentAnalysisSocialMedia.py`.

---
