# MedSense — Drug Condition Predictor

A machine learning web app that predicts patient conditions from drug reviews and recommends suitable drugs.

---

## What it does

- Takes a patient review or symptom description as input
- Predicts whether the condition is **Depression**, **High Blood Pressure**, or **Type 2 Diabetes**
- Recommends the best drugs for the predicted condition
- Shows real patient reviews for reference

---

## How to run

**Install dependencies:**
```bash
pip install streamlit pandas numpy scikit-learn scipy joblib
```

**Run the app:**
```bash
[streamlit run app.py](https://drug-predictor-8rvmmafmvqcunpumsmvdky.streamlit.app/)
```

Opens at `http://localhost:8501`



## Files needed

| File                    | Description              |

| `task6_dashboard.py.py` | Main web dashboard       |
| `tuned_model.pkl`       | Hist gradient boosting   |
| `tfidf_vectorizer.pkl`  | Text feature extractor   |
| `scaler.pkl`            | Numerical feature scaler |
| `label_encoder.pkl`     | Condition label encoder  |
| `train_cleaned.csv`     | Cleaned training data    |



## Model

- **Algorithm:** Linear SVM
- **Accuracy:** 94.8%
- **Features:** TF-IDF text (20,000) + 12 numerical features


## Built with

Python · scikit-learn · Streamlit · pandas · Jupyter Notebook

