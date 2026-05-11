"""
TASK 6 - MEDSENSE DRUG CONDITION PREDICTOR
Pure Streamlit - Zero HTML errors
RUN: streamlit run task6_dashboard.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize  import word_tokenize
from nltk.corpus    import stopwords
from nltk.stem      import WordNetLemmatizer
from textblob       import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)


# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "MedSense - Drug Condition Predictor",
    page_icon  = "💊",
    layout     = "centered",
    initial_sidebar_state = "collapsed",
)


# ────────────────────────────────────────────────────────────
# CSS - static only, no dynamic HTML
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; }

.block-container {
    max-width: 760px !important;
    padding-top: 1.5rem !important;
}
.stTextArea label { display: none !important; }
.stTextArea textarea {
    background: #1e1e2e !important;
    color: #cdd6f4 !important;
    border: 1px solid #45475a !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    pointer-events: auto !important;
    cursor: text !important;
}
.stTextArea textarea:focus {
    border-color: #89b4fa !important;
}
.stTextArea textarea::placeholder {
    color: #585b70 !important;
}
.stSlider label {
    color: #a6adc8 !important;
    font-size: 13px !important;
}
.stButton > button {
    background: #1e66f5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    padding: .65rem !important;
}
.stButton > button:hover {
    background: #1655d4 !important;
}
div[data-testid="metric-container"] {
    background: #1e1e2e !important;
    border: 1px solid #313244 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# CONSTANTS - keyword lists from TF-IDF training vocabulary
# ────────────────────────────────────────────────────────────
SAVE_DIR = r"C:\Users\Private\OneDrive\Desktop\cleaned"

ICONS = {
    "Depression":          "🧠",
    "High Blood Pressure": "❤️",
    "Type 2 Diabetes":     "🩸",
}
COLORS = {
    "Depression":          "#7C6AF0",
    "High Blood Pressure": "#F06A6A",
    "Type 2 Diabetes":     "#43C69A",
}

SIDE_EFFECT_WORDS = [
    "nausea","dizzy","dizziness","headache","fatigue","insomnia",
    "vomiting","rash","anxiety","weight gain","weight loss",
    "drowsy","tired","constipation","diarrhea","dry mouth",
    "blurred vision","chest pain",
]
EFFECTIVENESS_WORDS = [
    "works","helped","effective","relief","improved","better",
    "recommend","great","amazing","miracle","cure","cured",
    "useless","ineffective","worse",
]
NEGATION_WORDS = [
    "not","no","never","without",
    "don't","doesn't","didn't","couldn't",
]

# Built from actual TF-IDF top 60 training vocabulary
DEPRESSION_WORDS = [
    "depression","depressed","anxiety","feeling","felt","sleep",
    "mood","antidepressant","antidepressants","zoloft","lexapro",
    "wellbutrin","effexor","pristiq","prozac","sertraline",
    "fluoxetine","citalopram","venlafaxine","mirtazapine",
    "escitalopram","bupropion","duloxetine","cymbalta","paxil",
    "psychiatrist","therapist","hopeless","worthless","suicidal",
    "panic","phobia","crying","sadness","helpless","emptiness",
    "tearful","unmotivated","lonely","dose","doctor","medication",
]
HBP_WORDS = [
    "pressure","blood","heart","cough","high","lower","dizziness",
    "rate","normal","pill","pain","headache","leg","med","dose",
    "doctor","medication","blood pressure","high blood","lisinopril",
    "benicar","bystolic","diovan","amlodipine","losartan","atenolol",
    "metoprolol","valsartan","nifedipine","hydrochlorothiazide",
    "ramipril","carvedilol","enalapril","diuretic","hypertension",
    "hypertensive","systolic","diastolic","nosebleed","nosebleeds",
    "stroke","palpitation","palpitations","mmhg","antihypertensive",
    "cardiovascular","lightheaded","lightheadedness","throbbing",
    "racing","shortness","breath",
]
DIABETES_WORDS = [
    "sugar","injection","insulin","appetite","stomach","shot",
    "nausea","diarrhea","pound","lb","level","lost","weight",
    "eat","morning","infection","blood sugar","sugar level",
    "lost lb","weight loss","metformin","victoza","trulicity",
    "glucophage","januvia","jardiance","ozempic","glipizide",
    "invokana","farxiga","actos","glimepiride","sitagliptin",
    "empagliflozin","diabetes","diabetic","glucose","glycemic",
    "a1c","pancreas","neuropathy","hyperglycemia","thirsty",
    "thirst","urinate","urinating","urination","healing",
    "wounds","blurry","exhausted","frequent",
]

MIN_KEYWORD_SCORE = 1
MIN_MODEL_CONF    = 40.0


# ────────────────────────────────────────────────────────────
# LOAD ASSETS
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    return (
        joblib.load(os.path.join(SAVE_DIR, "final_tuned_model.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "label_encoder_condition.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "tfidf_chi2_selector.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "tfidf_pca.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "final_scaler.pkl")),
        joblib.load(os.path.join(SAVE_DIR, "final_scalar_features.pkl")),
    )

@st.cache_data(show_spinner="Loading data...")
def load_data():
    return pd.read_csv(os.path.join(SAVE_DIR, "features_extracted.csv"))

(model, le_cond, tfidf, tfidf_selector,
 tfidf_pca, final_scaler, final_features) = load_assets()

df          = load_data()
CLASS_NAMES = le_cond.classes_
vader       = SentimentIntensityAnalyzer()
lemmatizer  = WordNetLemmatizer()
ALL_STOP    = set(stopwords.words("english"))


# ────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"&#\d+;",        "'", text)
    text = re.sub(r"http\S+|www\S+", "",  text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in word_tokenize(text)
        if w not in ALL_STOP and len(w) > 2
    )


def keyword_override(raw_text):
    text_lower = raw_text.lower()
    words      = set(re.sub(r"[^a-z\s]", " ", text_lower).split())
    scores = {
        "Type 2 Diabetes": sum(
            1 for w in DIABETES_WORDS
            if (w in words if " " not in w else w in text_lower)
        ),
        "Depression": sum(
            1 for w in DEPRESSION_WORDS
            if (w in words if " " not in w else w in text_lower)
        ),
        "High Blood Pressure": sum(
            1 for w in HBP_WORDS
            if (w in words if " " not in w else w in text_lower)
        ),
    }
    best_cond  = max(scores, key=scores.get)
    best_score = scores[best_cond]
    if best_score >= 2:
        return best_cond, scores
    return None, scores


def is_relevant(kw_scores):
    return sum(kw_scores.values()) >= MIN_KEYWORD_SCORE


def build_feature_vector(text, rating=5):
    clean  = clean_text(text)
    words  = word_tokenize(clean) if clean else []
    tb     = TextBlob(text)
    vs     = vader.polarity_scores(text)
    tl     = text.lower()
    se     = sum(1 for w in SIDE_EFFECT_WORDS   if w in tl)
    ef     = sum(1 for w in EFFECTIVENESS_WORDS if w in tl)
    neg    = sum(1 for w in NEGATION_WORDS       if w in tl.split())
    rn     = rating / 10.0

    scalar = {
        "char_count":            len(text),
        "word_count":            len(words),
        "sentence_count":        max(1, text.count(".")),
        "avg_word_length":       float(np.mean([len(w) for w in words])) if words else 0.0,
        "unique_word_ratio":     len(set(words)) / len(words) if words else 0.0,
        "exclamation_count":     text.count("!"),
        "question_count":        text.count("?"),
        "capital_ratio":         sum(1 for c in text if c.isupper()) / max(1, len(text)),
        "flesch_reading_ease":   50.0,
        "gunning_fog":           10.0,
        "smog_index":            8.0,
        "vader_positive":        vs["pos"],
        "vader_negative":        vs["neg"],
        "vader_neutral":         vs["neu"],
        "vader_compound":        vs["compound"],
        "tb_polarity":           float(tb.sentiment.polarity),
        "tb_subjectivity":       float(tb.sentiment.subjectivity),
        "side_effect_count":     se,
        "effectiveness_count":   ef,
        "negation_count":        neg,
        "has_side_effects":      int(se > 0),
        "is_effective":          int(ef > 0),
        "review_year":           2024,
        "review_month":          6,
        "review_quarter":        2,
        "review_dayofweek":      1,
        "days_since_first_review": 0,
        "rating_normalized":     rn,
        "is_satisfied":          int(rating >= 7),
        "useful_log":            0.0,
        "is_useful":             0,
        "drugName_encoded":      -1,
        "condition_encoded":     -1,
        "drug_avg_rating":       0.0,
        "drug_review_count":     0,
        "drug_avg_sentiment":    0.0,
        "drug_avg_useful":       0.0,
        "drug_satisfaction_rate":0.0,
        "ner_org_count":         0,
        "ner_cardinal_count":    0,
        "ner_date_count":        0,
        "sentiment_x_effectiveness": vs["compound"] * ef,
        "sentiment_x_sideeffects":   vs["compound"] * se,
        "subjectivity_x_wordcount":  float(tb.sentiment.subjectivity) * len(words),
        "rating_sentiment_agreement":rn * vs["compound"],
        "useful_x_sentiment":    0.0,
        "drug_rep_x_rating":     0.0,
        "negation_x_sideeffects":neg * se,
        "readability_x_wordcount":50.0 * len(words),
        "pos_neg_ratio":         vs["pos"] / max(vs["neg"], 1e-6),
    }

    arr           = np.array([[scalar.get(f, 0.0) for f in final_features]])
    arr           = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    tfidf_vec     = tfidf.transform([clean])
    tfidf_sel     = tfidf_selector.transform(tfidf_vec)
    tfidf_pca_vec = tfidf_pca.transform(tfidf_sel.toarray())
    return np.hstack([final_scaler.transform(arr), tfidf_pca_vec])


def predict_condition(review_text, rating=5):
    _, kw_scores = keyword_override(review_text)

    if not is_relevant(kw_scores):
        return {
            "valid":   False,
            "message": "Input does not appear to describe symptoms related to Depression, High Blood Pressure, or Type 2 Diabetes.",
        }

    override_cond, kw_scores = keyword_override(review_text)
    if override_cond is not None:
        total      = max(sum(kw_scores.values()), 1)
        proba_dict = {c: round(kw_scores.get(c, 0) / total * 100, 2) for c in CLASS_NAMES}
        proba_dict[override_cond] = max(proba_dict[override_cond], 70.0)
        remainder  = 100.0 - proba_dict[override_cond]
        others     = [c for c in CLASS_NAMES if c != override_cond]
        for c in others:
            proba_dict[c] = round(remainder / len(others), 2)
        return {
            "valid":     True,
            "condition": override_cond,
            "confidence":proba_dict[override_cond],
            "proba":     proba_dict,
            "method":    "keyword",
            "message":   "",
        }

    X          = build_feature_vector(review_text, rating=rating)
    proba      = model.predict_proba(X)[0]
    proba_dict = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)}
    pred_label = max(proba_dict, key=proba_dict.get)
    confidence = proba_dict[pred_label]
    low_msg    = (
        f"Low confidence ({confidence:.1f}%). Try adding more specific symptoms or drug names."
        if confidence < MIN_MODEL_CONF else ""
    )
    return {
        "valid":     True,
        "condition": pred_label,
        "confidence":confidence,
        "proba":     proba_dict,
        "method":    "model",
        "message":   low_msg,
    }


def get_top_drugs(condition, top_n=5):
    cdf   = df[df["condition"] == condition]
    stats = (
        cdf.groupby("drugName")
        .agg(
            avg_rating   =("rating",      "mean"),
            review_count =("rating",      "count"),
            avg_useful   =("usefulCount", "mean"),
        )
        .reset_index()
    )
    C = stats["avg_rating"].mean()
    m = 10
    stats["weighted_rating"] = (
        (stats["review_count"] * stats["avg_rating"] + m * C)
        / (stats["review_count"] + m)
    ).round(2)
    return (
        stats
        .sort_values("weighted_rating", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_sample_reviews(condition, top_n=3):
    cdf = (
        df[df["condition"] == condition]
        .sort_values("usefulCount", ascending=False)
    )
    out = []
    for _, row in cdf.head(top_n).iterrows():
        out.append({
            "drug":   row["drugName"],
            "rating": round(float(row["rating"]), 1),
            "useful": int(row["usefulCount"]),
            "review": str(row["review"])[:300] + "...",
        })
    return out


def confidence_label(c):
    if c >= 90:   return "🟢 Very High"
    elif c >= 75: return "🟡 High"
    elif c >= 60: return "🟠 Moderate"
    else:         return "🔴 Low — try adding more detail"


# ────────────────────────────────────────────────────────────
# UI — HEADER
# ────────────────────────────────────────────────────────────
st.markdown("## 💊 MedSense — Drug Condition Predictor")
st.caption(
    "Predicts: 🧠 Depression  |  ❤️ High Blood Pressure  |  🩸 Type 2 Diabetes  \n"
    "💡 Include specific symptoms and drug/condition keywords for accurate predictions."
)
st.divider()


# ────────────────────────────────────────────────────────────
# UI — INPUT
# ────────────────────────────────────────────────────────────
review_text = st.text_area(
    label            = "Describe your symptoms",
    height           = 140,
    placeholder      = (
        "Describe your symptoms in detail...\n\n"
        "Example: My blood sugar levels are very high. "
        "I feel extremely thirsty and urinate frequently. "
        "My doctor prescribed metformin to control my blood glucose."
    ),
    key              = "symptom_input",
)

rating = st.slider(
    "Self-rated severity (1 = mild, 10 = severe)",
    min_value=1, max_value=10, value=5, step=1,
)

col1, col2 = st.columns([4, 1])
with col1:
    predict_btn = st.button("⚡  Predict Condition", use_container_width=True)
with col2:
    st.button("🗑  Clear", use_container_width=True)


# ────────────────────────────────────────────────────────────
# UI — VALIDATION
# ────────────────────────────────────────────────────────────
if predict_btn:
    if not review_text or len(review_text.strip()) < 15:
        st.warning("⚠️ Please describe your symptoms in at least 15 characters.")
        st.stop()


# ────────────────────────────────────────────────────────────
# UI — PREDICTION & RESULTS
# ────────────────────────────────────────────────────────────
if predict_btn and review_text.strip():

    with st.spinner("Analyzing symptoms..."):
        try:
            result = predict_condition(review_text, rating=rating)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # Rejected input
    if not result["valid"]:
        st.warning(f"🚫 **Cannot predict.**  \n{result['message']}")
        st.info(
            "This system only predicts:  \n"
            "🧠 Depression  |  ❤️ High Blood Pressure  |  🩸 Type 2 Diabetes"
        )
        st.stop()

    cond   = result["condition"]
    conf   = result["confidence"]
    proba  = result["proba"]
    method = result["method"]
    msg    = result["message"]
    icon   = ICONS[cond]

    # Low confidence warning
    if msg:
        st.warning(f"⚠️ {msg}")

    st.divider()

    # ── 1. PREDICTED CONDITION ────────────────────────────────
    st.markdown(f"### {icon} Predicted Condition")

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"## {cond}")
        badge = "🔑 keyword match" if method == "keyword" else "🤖 model prediction"
        st.caption(f"{badge}  |  {confidence_label(conf)}")
    with right:
        st.metric(label="Confidence", value=f"{conf:.1f}%")

    st.divider()

    # ── 2. PROBABILITY BREAKDOWN ──────────────────────────────
    st.markdown("### 📊 Probability Breakdown")

    sorted_proba = sorted(proba.items(), key=lambda x: -x[1])
    for c, p in sorted_proba:
        c1, c2 = st.columns([6, 1])
        with c1:
            st.caption(f"{ICONS[c]}  {c}")
            st.progress(int(p))
        with c2:
            st.markdown(f"**{p:.1f}%**")

    st.divider()

    # ── 3. DRUG RECOMMENDATIONS ───────────────────────────────
    st.markdown(f"### 💊 Recommended Drugs — {cond}")
    st.caption("Ranked by Bayesian weighted rating — penalises drugs with very few reviews")

    top_drugs = get_top_drugs(cond)

    table_df = pd.DataFrame({
        "Rank":            [f"#{i+1}" for i in range(len(top_drugs))],
        "Drug Name":       top_drugs["drugName"].tolist(),
        "Weighted Rating": [f"{r:.1f} / 10" for r in top_drugs["weighted_rating"]],
        "Total Reviews":   [int(r) for r in top_drugs["review_count"]],
        "Avg Useful":      [f"{r:.1f}" for r in top_drugs["avg_useful"]],
    })

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── 4. PATIENT REVIEWS ────────────────────────────────────
    st.markdown(f"### 📝 Real Patient Reviews — {cond}")
    st.caption("Most helpful reviews from patients with the same condition")

    reviews = get_sample_reviews(cond)
    for r in reviews:
        with st.expander(
            f"**{r['drug']}**  |  "
            f"Rating: {r['rating']} / 10  |  "
            f"👍 {r['useful']} found helpful"
        ):
            st.write(r["review"])