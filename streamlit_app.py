"""
streamlit_app.py — Streamlit Frontend for the Sentiment Analysis API
=====================================================================

A simple web interface that connects to the running FastAPI backend
to demonstrate sentiment predictions interactively.

Prerequisites:
  1. Run the FastAPI server first:   uvicorn app.main:app --reload
  2. Then run this frontend:         streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
PREDICT_URL = f"{API_BASE_URL}/predict"
BATCH_URL = f"{API_BASE_URL}/predict/batch"
HEALTH_URL = f"{API_BASE_URL}/health"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Sentiment Analysis")
st.caption("Powered by TF-IDF + Logistic Regression · Trained on IMDB Movie Reviews")

# ---------------------------------------------------------------------------
# API health check — show a warning banner if the backend is not reachable
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


if not check_api_health():
    st.error(
        "⚠️ Cannot reach the FastAPI backend at `http://localhost:8000`. "
        "Please start it first with: `uvicorn app.main:app --reload`",
        icon="🚨",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs — Single Prediction | Batch Prediction
# ---------------------------------------------------------------------------

tab_single, tab_batch = st.tabs(["📝 Single Prediction", "📋 Batch Prediction"])


# ── Tab 1: Single Prediction ────────────────────────────────────────────────

with tab_single:
    st.subheader("Predict sentiment for a single review")

    text_input = st.text_area(
        label="Enter a review:",
        placeholder="e.g. I absolutely loved this movie. The acting was superb!",
        height=150,
    )

    if st.button("Analyse", type="primary", key="single_btn"):
        if not text_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        PREDICT_URL,
                        json={"text": text_input},
                        timeout=10,
                    )
                    response.raise_for_status()
                    data = response.json()

                    sentiment = data["sentiment"]
                    confidence = data["confidence"]

                    # Display result
                    st.divider()
                    if sentiment == "positive":
                        st.success(f"**Sentiment: Positive** 😊", icon="✅")
                    else:
                        st.error(f"**Sentiment: Negative** 😞", icon="❌")

                    st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
                    st.progress(confidence)

                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")


# ── Tab 2: Batch Prediction ─────────────────────────────────────────────────

with tab_batch:
    st.subheader("Predict sentiment for multiple reviews at once")
    st.caption("Enter one review per line.")

    batch_input = st.text_area(
        label="Enter reviews (one per line):",
        placeholder=(
            "I loved every moment of this film!\n"
            "Terrible acting, complete waste of time.\n"
            "It was an average movie, nothing special."
        ),
        height=200,
    )

    if st.button("Analyse Batch", type="primary", key="batch_btn"):
        texts = [line.strip() for line in batch_input.splitlines() if line.strip()]

        if not texts:
            st.warning("Please enter at least one review.")
        else:
            with st.spinner(f"Analyzing {len(texts)} review(s)..."):
                try:
                    response = requests.post(
                        BATCH_URL,
                        json={"texts": texts},
                        timeout=30,
                    )
                    response.raise_for_status()
                    predictions = response.json()["predictions"]

                    st.divider()
                    st.write(f"**Results for {len(predictions)} review(s):**")

                    for pred in predictions:
                        sentiment = pred["sentiment"]
                        confidence = pred["confidence"]
                        icon = "✅" if sentiment == "positive" else "❌"
                        label = "Positive" if sentiment == "positive" else "Negative"

                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"{icon} **{label}** — {pred['text'][:120]}{'...' if len(pred['text']) > 120 else ''}")
                            with col2:
                                st.metric("Confidence", f"{confidence * 100:.1f}%")

                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

# ---------------------------------------------------------------------------
# Sidebar — about
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This frontend connects to a **FastAPI** backend running locally.

        **Model:** TF-IDF + Logistic Regression  
        **Dataset:** IMDB Movie Reviews (50,000 samples)  
        **Accuracy:** ~90.45% on test set

        **API Endpoints:**
        - `GET /health`
        - `POST /predict`
        - `POST /predict/batch`

        View the full API docs at  
        [localhost:8000/docs](http://localhost:8000/docs)
        """
    )
    st.divider()
    # Live API status indicator
    if check_api_health():
        st.success("API: Online ✅")
    else:
        st.error("API: Offline ❌")
