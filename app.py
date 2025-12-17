import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import analyze_sentiment

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="centered"
)

# ------------------ App Header ------------------
st.title("🧠 AI Sentiment Analyzer")
st.subheader("Understand emotions in text using NLP")
st.write(
    "This app uses a Transformer-based NLP model to classify text as "
    "**Positive** or **Negative** with confidence scores."
)

# ------------------ Session State ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ User Input ------------------
text_input = st.text_area(
    "✍🏽 Enter text to analyze",
    placeholder="e.g. I absolutely love this product!",
    height=120,
    key="text_input"
)
# ------------------ CSV Upload ------------------
st.markdown("---")
st.subheader("📂 Analyze from CSV file")
uploaded_file = st.file_uploader(
    "Upload a CSV file with a 'text' column for batch sentiment analysis",
    type=["csv"]
)
if uploaded_file is not None:
    df_uploaded = pd.read_csv(
    uploaded_file,
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"
)
    st.write("📄 Preview of uploaded data:")
    st.dataframe(df_uploaded.head())

    text_column = st.selectbox(
        "Select the column that contains text",
        df_uploaded.columns
    )

    if st.button("Analyze CSV", key="analyze_csv_btn"):
        with st.spinner("Analyzing CSV file..."):
            for text in df_uploaded[text_column].astype(str):
                label, score = analyze_sentiment(text)

                st.session_state.history.append({
                    "Text": text,
                    "Sentiment": label,
                    "Confidence": round(score, 2)
                })

        st.success("✅ CSV analysis completed!")

# ------------------ Analyze Button ------------------
if st.button("Analyze", key="analyze_btn"):
    if text_input:
        with st.spinner("Analyzing sentiment..."):
            label, score = analyze_sentiment(text_input)

        st.session_state.history.append({
            "Text": text_input,
            "Sentiment": label,
            "Confidence": round(score, 2)
        })

        if label == "POSITIVE":
            st.success("😄 Positive sentiment detected!")
        else:
            st.error("😠 Negative sentiment detected!")

        confidence_percent = int(score * 100)

        st.write("**Confidence Level**")
        st.progress(confidence_percent)

        emoji = "🔥" if confidence_percent > 80 else "🙂" if confidence_percent > 60 else "⚠️"
        st.caption(f"{emoji} {confidence_percent}% confidence in this prediction")

    else:
        st.warning("Please enter some text first.")

# ------------------ Visualization ------------------
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    st.download_button(
        label="⬇️ Download analysis as CSV",
        data=df.to_csv(index=False),
        file_name="sentiment_analysis_history.csv",
        mime="text/csv"
    )

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

