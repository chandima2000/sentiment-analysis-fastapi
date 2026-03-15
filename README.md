# Sentiment Analysis API

A REST API that predicts the **sentiment** of a given text (`positive` or `negative`) with a confidence score. Built with **FastAPI** and a **TF-IDF + Logistic Regression** model trained on the IMDB movie reviews dataset.

---

## Requirements

- Python **3.10+**
- pip

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/chandima2000/sentiment-analysis-fastapi.git
cd sentiment-analysis-fastapi
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the training script **once** before starting the server. It reads the IMDB dataset, trains the classifier, saves the model, and prints an evaluation report.

> **Note:** Download the dataset from the following link, create a `data/` folder, place the CSV inside it and rename it to `IMDB_Dataset.csv` before running.
> 📎 [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

```bash
python train.py
```

**Expected output:**

```
=======================================================
  Sentiment Analysis — Training Pipeline
=======================================================
Loading dataset from: data\IMDB_Dataset.csv
Loaded 50,000 reviews  (25,000 positive | 25,000 negative)

Splitting data  (80% train / 20% test, stratified)...
   Train: 40,000 samples   |   Test: 10,000 samples
Training TF-IDF + Logistic Regression pipeline...
Training complete.

Evaluation Report (Test Set):
-------------------------------------------------------
              precision    recall  f1-score   support

    negative     0.9145    0.8924    0.9033      5000
    positive     0.8949    0.9166    0.9056      5000

    accuracy                         0.9045     10000
   macro avg     0.9047    0.9045    0.9045     10000
weighted avg     0.9047    0.9045    0.9045     10000

Overall Accuracy: 0.9045

Evaluation report saved → result\evaluation_report.txt
Model saved → model\sentiment_pipeline.pkl

Done! Start the API with:
   uvicorn app.main:app --reload
```

---

## Start the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at **`http://127.0.0.1:8000`**

Interactive API docs (Swagger UI): **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## Run the Streamlit Frontend

A simple web interface is included for interactive demonstrations.

> **Note:** The FastAPI server must be running before launching the frontend.

Open a **second terminal** and run:

```bash
streamlit run streamlit_app.py
```

The frontend will open at **`http://localhost:8501`** and includes:
- **Single Prediction tab** — enter a review, get sentiment + confidence bar
- **Batch Prediction tab** — enter multiple reviews (one per line), see all results
- **Sidebar** — live API status indicator and model info

---

## API Usage

### Health Check

**Windows (PowerShell)**
```powershell
curl.exe http://localhost:8000/health
```

**Linux / macOS**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

---

### Predict Sentiment — Single Text

**Windows (PowerShell)**
```powershell
$body = '{"text": "I absolutely loved this movie. The acting was superb!"}'
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d $body
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie. The acting was superb!"}'
```

**Response:**
```json
{
  "text": "I absolutely loved this movie. The acting was superb!",
  "sentiment": "positive",
  "confidence": 0.9614
}
```

---

### Predict Sentiment — Batch

**Windows (PowerShell)**
```powershell
$body = '{"texts": ["Amazing film, highly recommend!", "Terrible waste of time.", "It was okay, nothing special."]}'
curl.exe -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d $body
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing film, highly recommend!", "Terrible waste of time.", "It was okay, nothing special."]}'
```

**Response:**
```json
{
  "predictions": [
    {"text": "Amazing film, highly recommend!", "sentiment": "positive", "confidence": 0.9922},
    {"text": "Terrible waste of time.", "sentiment": "negative", "confidence": 0.9973},
    {"text": "It was okay, nothing special.", "sentiment": "negative", "confidence": 0.808}
  ]
}
```

**Empty list — returns HTTP 422:**

**Windows (PowerShell)**
```powershell
$body = '{"texts": []}'
curl.exe -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d $body
```

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": []}'
```

---

## Project Structure

```
sentiment-analysis-fastapi/
├── app/
│   ├── __init__.py        # Package marker
│   ├── main.py            # FastAPI app — routes & lifespan model loader
│   ├── model.py           # ML pipeline loading and inference logic
│   └── schemas.py         # Pydantic request/response models
├── data/                  # Place IMDB_Dataset.csv here (not committed)
├── model/
│   └── sentiment_pipeline.pkl  # Saved model (generated by train.py)
├── result/
│   └── evaluation_report.txt   # Evaluation metrics (generated by train.py)
├── train.py               # Training script
├── streamlit_app.py       # Streamlit web frontend
├── requirements.txt       # Pinned dependencies
└── README.md
```

---

## Approach

### Dataset Selection

Two datasets were evaluated for training the sentiment classifier: the **Twitter US Airline Sentiment** dataset and the **IMDb Movie Review** dataset. While the Twitter dataset contains three sentiment classes (positive, neutral, negative), it consists of short and noisy social media text. The IMDb dataset contains 50,000 well-structured movie reviews labelled as positive or negative, making it more suitable for training a robust sentiment classifier. Therefore, the **IMDb dataset was selected** for this implementation.

### Model Selection

Two models were evaluated:

- **TF-IDF + Multinomial Naive Bayes** — fast to train and works well with small datasets, but makes an unrealistic independence assumption between words and produces poorly calibrated probability scores.
- **TF-IDF + Logistic Regression** *(selected)* — slightly slower to train but produces **well-calibrated confidence scores** via `predict_proba`, makes no false independence assumptions, and consistently achieves higher accuracy on the IMDb benchmark.

I chose **TF-IDF + Logistic Regression** because it trains in seconds on CPU, requires no GPU, and produces well-calibrated confidence scores via `predict_proba`, making the returned `confidence` value genuinely meaningful. The model is wrapped in a scikit-learn **Pipeline** object so the exact same TF-IDF preprocessing (vocabulary, IDF weights) applied during training is always applied at inference, eliminating train-serve skew. I used **bigrams** (`ngram_range=(1, 2)`) to capture two-word sentiment signals like "not good" or "highly recommend" that unigrams alone would miss. With more time, I would fine-tune a **DistilBERT** transformer model for higher accuracy (~93% vs ~89%), add a `pytest` suite covering edge cases, and containerise the service with **Docker** for portable deployment.


---

## References

**TF-IDF Vectorizer**
1. [scikit-learn — TfidfVectorizer (Official Docs)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [How to Use sklearn's TfidfVectorizer for Text Feature Extraction](https://medium.com/@masudowolabi/how-to-use-sklearns-tfidfvectorizer-for-text-feature-extraction-in-model-testing-e1221fd274f8)

**Logistic Regression**

3. [scikit-learn — LogisticRegression (Official Docs)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
4. [Understanding Logistic Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
5. [Understanding Logistic Regression in Python — DataCamp](https://www.datacamp.com/tutorial/understanding-logistic-regression-python)
