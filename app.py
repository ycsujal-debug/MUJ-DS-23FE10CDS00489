"""
ReviewRadar: Interactive Streamlit Forensic NLP Dashboard.
Combines classical NLP diagnostics (Tokenization, Normalization, Penn Treebank POS Ratios,
Laplace & Good-Turing smoothed N-Gram Perplexity) with an LLM-powered Expert Fraud Analyst.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.pipeline import ReviewRadarPipeline
from src.llm.structured_parser import FraudClassification, RiskLevel, SentimentType

# Page Configuration
st.set_page_config(
    page_title="ReviewRadar | E-Commerce Fake Review & Bot Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .verdict-box-genuine {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .verdict-box-bot {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .verdict-box-spam {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> ReviewRadarPipeline:
    """Initialize and cache the ReviewRadar pipeline singleton."""
    return ReviewRadarPipeline()


pipeline = get_pipeline()

# Preset Sample Reviews
PRESET_REVIEWS = {
    "Select a preset sample...": "",
    "1. Genuine Organic Review (Detailed functional feedback)": (
        "I purchased this coffee maker three weeks ago for our apartment kitchen. "
        "The stainless steel thermal carafe keeps the brew hot for over three hours without burning the taste. "
        "The water reservoir is easy to detach and refill, and the buttons feel sturdy. "
        "Only minor downside is the power cord is slightly short, but otherwise it delivers rich espresso consistently."
    ),
    "2. Machine-Generated Bot (Repetitive template pattern)": (
        "Good product fast shipping. Great quality highly recommend. Good product fast shipping. "
        "Great quality highly recommend. Good product fast shipping. A++ seller."
    ),
    "3. Incentivized Promotional Spam (Superlative flooding & elongation)": (
        "OMGGGGG this is the absolute b3st pr0duct evvvver in the entire world!!!!!! "
        "Super duper amaaaazing quality, insanely fast delivery, 100% must buy right now guys!!!! "
        "Best purchase of my life, totally incredible, amazing, awesome, fantastic 5stars!!!"
    ),
    "4. Garbled Keyword Stuffing Spam (High entropy & OOV anomaly)": (
        "Ultra quantum crypto bluetooth discount deals 50% off best magic solution buy now "
        "cheap wholesale luxury free shipping miracle remedy guaranteed."
    )
}

# Sidebar Controls
st.sidebar.title("🛡️ ReviewRadar Config")
st.sidebar.markdown("**Forensic NLP Engine Settings**")

smoothing_option = st.sidebar.selectbox(
    "N-Gram Smoothing Method",
    options=["Laplace (Add-1)", "Good-Turing (Katz Backoff)"],
    index=0,
    help="Select the statistical smoothing algorithm used to estimate N-gram transition probabilities."
)
smoothing_key = "laplace" if "Laplace" in smoothing_option else "good_turing"

provider_option = st.sidebar.selectbox(
    "LLM Fraud Analyst Provider",
    options=["Heuristic Offline Engine (Deterministic)", "Google Gemini (REST API)", "OpenAI (REST API)"],
    index=0,
    help="Choose whether to invoke live cloud LLM APIs or use the offline computational linguistics rule engine."
)

if "Gemini" in provider_option:
    llm_key = "gemini"
    gemini_api_key = st.sidebar.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
elif "OpenAI" in provider_option:
    llm_key = "openai"
    openai_api_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    llm_key = "heuristic"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Syllabus & Algorithm Grounding")
st.sidebar.markdown("""
- **Text Normalization**: Regex-based elongation reduction, leetspeak translation, punctuation sanitization.
- **Penn Treebank POS Profiler**: Ratios of Modifiers `(JJ+RB)` to Content `(NN+VB)`, Superlative density, Lexical Diversity `(TTR)`.
- **N-Gram Perplexity**: Bigram/Trigram models with Laplace & Good-Turing smoothing for entropy anomaly detection.
- **LLM Fraud Analyst**: Interprets the statistical footprint into an authoritative forensic verdict.
""")

# Main Title Header
st.markdown('<div class="main-title">🛡️ ReviewRadar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">E-Commerce Fake Review & Bot Detection System | Classical NLP + LLM Forensics</div>', unsafe_allow_html=True)

# Application Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Review Inspector",
    "📊 Batch Review Screener",
    "🧪 Classical NLP Laboratory"
])

# ==========================================
# TAB 1: SINGLE REVIEW INSPECTOR
# ==========================================
with tab1:
    col_input, col_presets = st.columns([3, 2])
    with col_presets:
        selected_preset = st.selectbox("Load Sample Review Preset:", list(PRESET_REVIEWS.keys()))

    preset_text = PRESET_REVIEWS.get(selected_preset, "")
    review_input = st.text_area(
        "Enter E-Commerce Customer Review:",
        value=preset_text if preset_text else "Enter or paste a customer review here to analyze its linguistic authenticity...",
        height=140
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked or (preset_text and len(preset_text) > 10):
        if not review_input.strip() or review_input.startswith("Enter or paste"):
            st.warning("Please enter a valid review text.")
        else:
            with st.spinner("Executing Classical NLP Pipeline & LLM Fraud Analysis..."):
                report = pipeline.analyze(
                    review_text=review_input,
                    smoothing=smoothing_key,
                    llm_provider=llm_key
                )

            diag = report.diagnostics
            verdict = report.verdict

            st.markdown("---")

            # Top Verdict Banner
            if verdict.classification == FraudClassification.GENUINE:
                box_class = "verdict-box-genuine"
                verdict_icon = "✅"
                color_hex = "#10B981"
            elif verdict.classification == FraudClassification.MACHINE_GENERATED_BOT:
                box_class = "verdict-box-bot"
                verdict_icon = "🤖"
                color_hex = "#F59E0B"
            else:
                box_class = "verdict-box-spam"
                verdict_icon = "🚨"
                color_hex = "#EF4444"

            st.markdown(f"""
            <div class="{box_class}">
                <h3 style="margin-top:0; color:{color_hex};">{verdict_icon} Verdict: {verdict.classification.value.replace('_', ' ')}</h3>
                <p style="font-size:1.05rem; margin-bottom:0.5rem;"><strong>Linguistic Explanation:</strong> {verdict.linguistic_explanation}</p>
                <div style="margin-top:0.6rem;">
                    <span class="metric-badge" style="background-color:#E2E8F0; color:#334155;">Confidence: {verdict.confidence_score * 100:.1f}%</span>
                    <span class="metric-badge" style="background-color:#E2E8F0; color:#334155;">Risk Level: {verdict.fraud_risk_level.value}</span>
                    <span class="metric-badge" style="background-color:#E2E8F0; color:#334155;">Sentiment: {verdict.product_sentiment.sentiment.value} ({verdict.product_sentiment.sentiment_score:+.2f})</span>
                    <span class="metric-badge" style="background-color:#E2E8F0; color:#334155;">Processing Time: {report.processing_time_ms} ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    label="Modifier-to-Content Ratio",
                    value=f"{diag.modifier_to_content_ratio:.2f}",
                    delta="Spam > 0.65" if diag.modifier_to_content_ratio > 0.65 else "Healthy (0.25-0.55)",
                    delta_color="inverse" if diag.modifier_to_content_ratio > 0.65 else "normal"
                )
            with m2:
                st.metric(
                    label=f"Bigram Perplexity ({smoothing_key.title()})",
                    value=f"{diag.bigram_perplexity_laplace if smoothing_key=='laplace' else diag.bigram_perplexity_good_turing:.1f}",
                    delta="Bot < 20 / Spam > 400" if diag.bigram_perplexity_laplace < 20 or diag.bigram_perplexity_laplace > 400 else "Natural Range (30-250)",
                    delta_color="inverse" if diag.bigram_perplexity_laplace < 20 or diag.bigram_perplexity_laplace > 400 else "normal"
                )
            with m3:
                st.metric(
                    label="Lexical Diversity (TTR)",
                    value=f"{diag.type_token_ratio:.2f}",
                    delta="Low < 0.45" if diag.type_token_ratio < 0.45 else "Diverse (> 0.50)",
                    delta_color="inverse" if diag.type_token_ratio < 0.45 else "normal"
                )
            with m4:
                st.metric(
                    label="Normalization Fixes",
                    value=f"{diag.elongations_fixed + diag.excessive_punct_fixed + diag.leetspeak_fixed}",
                    delta=f"{diag.elongations_fixed} Elong | {diag.leetspeak_fixed} Leet | {diag.excessive_punct_fixed} Punct",
                    delta_color="off"
                )

            # Tabs for Detailed Breakdown
            d_tab1, d_tab2, d_tab3 = st.tabs([
                "📊 Classical NLP Footprint",
                "⚖️ Forensic Statistical Evidence",
                "📝 Normalization & Token Diff"
            ])

            with d_tab1:
                c_col1, c_col2 = st.columns(2)

                with c_col1:
                    st.subheader("Penn Treebank POS Category Breakdown")
                    pos_df = pd.DataFrame({
                        "Category": ["Adjectives (JJ)", "Adverbs (RB)", "Nouns (NN)", "Verbs (VB)", "Superlatives (JJS/RBS)"],
                        "Count": [
                            sum(diag.pos_tag_frequencies.get(t, 0) for t in ["JJ", "JJR", "JJS"]),
                            sum(diag.pos_tag_frequencies.get(t, 0) for t in ["RB", "RBR", "RBS"]),
                            sum(diag.pos_tag_frequencies.get(t, 0) for t in ["NN", "NNS", "NNP", "NNPS"]),
                            sum(diag.pos_tag_frequencies.get(t, 0) for t in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]),
                            sum(diag.pos_tag_frequencies.get(t, 0) for t in ["JJS", "RBS"]),
                        ]
                    })
                    fig_pos = px.bar(
                        pos_df,
                        x="Category",
                        y="Count",
                        color="Category",
                        color_discrete_sequence=["#3B82F6", "#60A5FA", "#10B981", "#34D399", "#F59E0B"],
                        title="Syntactic Distribution of Review Tokens"
                    )
                    fig_pos.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_pos, use_container_width=True)

                with c_col2:
                    st.subheader("N-Gram Perplexity Comparison")
                    pp_df = pd.DataFrame({
                        "Smoothing Method": ["Laplace (Add-1)", "Good-Turing (Katz)"],
                        "Perplexity Score": [diag.bigram_perplexity_laplace, diag.bigram_perplexity_good_turing]
                    })
                    fig_pp = px.bar(
                        pp_df,
                        x="Smoothing Method",
                        y="Perplexity Score",
                        color="Smoothing Method",
                        color_discrete_sequence=["#8B5CF6", "#EC4899"],
                        title="Statistical Uncertainty / Perplexity by Smoothing"
                    )
                    fig_pp.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_pp, use_container_width=True)

                if diag.sentence_perplexities and len(diag.sentence_perplexities) > 1:
                    st.subheader("Per-Sentence Perplexity Trajectory")
                    sent_df = pd.DataFrame({
                        "Sentence Index": [f"Sent {i+1}" for i in range(len(diag.sentence_perplexities))],
                        "Perplexity": diag.sentence_perplexities,
                        "Sentence Text": diag.sentences[:len(diag.sentence_perplexities)]
                    })
                    fig_sent = px.line(
                        sent_df,
                        x="Sentence Index",
                        y="Perplexity",
                        markers=True,
                        hover_data=["Sentence Text"],
                        title="Sentence-by-Sentence Entropy Dynamics"
                    )
                    fig_sent.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_sent, use_container_width=True)

            with d_tab2:
                st.subheader("Forensic Evidence Cited by LLM Analyst")
                if verdict.statistical_evidence:
                    ev_data = [
                        {
                            "Metric Name": item.metric_name,
                            "Observed Value": item.observed_value,
                            "Baseline Range": item.baseline_range,
                            "Forensic Interpretation": item.interpretation
                        }
                        for item in verdict.statistical_evidence
                    ]
                    st.dataframe(pd.DataFrame(ev_data), use_container_width=True)
                else:
                    st.info("No specific anomaly flags cited.")

                col_flags, col_sentiment = st.columns(2)
                with col_flags:
                    st.markdown("**Triggered Anomaly Flags:**")
                    if verdict.anomaly_flags:
                        for flag in verdict.anomaly_flags:
                            st.markdown(f"- 🚩 `{flag}`")
                    else:
                        st.markdown("✅ *No statistical anomaly flags triggered.*")

                with col_sentiment:
                    st.markdown("**Product Feature Extraction:**")
                    if verdict.product_sentiment.key_features_mentioned:
                        for feat in verdict.product_sentiment.key_features_mentioned:
                            st.markdown(f"- 🏷️ **{feat}**")
                    else:
                        st.markdown("*(No genuine product features identified)*")

            with d_tab3:
                st.subheader("Text Normalization Transformation")
                col_r, col_n = st.columns(2)
                with col_r:
                    st.markdown("**Raw Input Text:**")
                    st.code(diag.raw_text, language="text")
                with col_n:
                    st.markdown("**Normalized Text:**")
                    st.code(diag.normalized_text, language="text")

                st.markdown("**Extracted Word Tokens:**")
                st.write(diag.tokens)

# ==========================================
# TAB 2: BATCH REVIEW SCREENER
# ==========================================
with tab2:
    st.subheader("📊 Batch E-Commerce Review Fraud Screener")
    st.markdown("Upload a CSV file containing reviews or load our demonstration evaluation dataset.")

    col_upload, col_demo = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV (Must have a 'review' or 'text' column):", type=["csv"])

    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        load_demo_batch = st.button("📂 Load Demo Batch (6 Reviews)", use_container_width=True)

    batch_texts: List[str] = []
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            target_col = next((c for c in ["review", "text", "Review", "Text", "content"] if c in df_in.columns), None)
            if target_col:
                batch_texts = df_in[target_col].dropna().astype(str).tolist()
                st.success(f"Loaded {len(batch_texts)} reviews from `{uploaded_file.name}`.")
            else:
                st.error(f"Could not find a 'review' or 'text' column in CSV. Found columns: {list(df_in.columns)}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif load_demo_batch:
        batch_texts = [
            "The battery life on this wireless headphone exceeds my expectations and lasts two full days.",
            "Good product fast shipping. Great quality highly recommend. Good product fast shipping.",
            "OMGGGGG this is the absolute b3st pr0duct evvvver in the entire world!!!!!! 5stars!!!",
            "The coffee maker brews piping hot coffee in two minutes. The stainless carafe is easy to clean.",
            "Super duper awesome magic gadget buy now 100% discount free gift amazing fantastic!",
            "I have used this cordless drill for weekend DIY projects. The torque is powerful and keyless chuck is solid."
        ]
        st.info(f"Loaded {len(batch_texts)} demo reviews.")

    if batch_texts:
        if st.button("⚡ Run Batch Fraud Audit", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_results = []

            for i, text in enumerate(batch_texts):
                status_text.text(f"Auditing Review {i+1} of {len(batch_texts)}...")
                report = pipeline.analyze(text, smoothing=smoothing_key, llm_provider=llm_key)
                batch_results.append({
                    "Review ID": report.review_id,
                    "Raw Review": text[:80] + "..." if len(text) > 80 else text,
                    "Verdict": report.verdict.classification.value,
                    "Confidence": report.verdict.confidence_score,
                    "Risk Level": report.verdict.fraud_risk_level.value,
                    "Modifier Ratio": report.diagnostics.modifier_to_content_ratio,
                    "Laplace PP": report.diagnostics.bigram_perplexity_laplace,
                    "Lexical TTR": report.diagnostics.type_token_ratio,
                    "Sentiment": report.verdict.product_sentiment.sentiment.value,
                    "Flags Count": len(report.verdict.anomaly_flags)
                })
                progress_bar.progress((i + 1) / len(batch_texts))

            status_text.text("Batch Audit Complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            df_batch = pd.DataFrame(batch_results)

            st.markdown("### Batch Audit Summary")
            b1, b2, b3 = st.columns(3)
            with b1:
                genuine_pct = (df_batch["Verdict"] == "GENUINE").mean() * 100
                st.metric("Genuine Reviews", f"{genuine_pct:.1f}%")
            with b2:
                bot_pct = (df_batch["Verdict"] == "MACHINE_GENERATED_BOT").mean() * 100
                st.metric("Bot Templates", f"{bot_pct:.1f}%")
            with b3:
                spam_pct = (df_batch["Verdict"] == "INCENTIVIZED_SPAM").mean() * 100
                st.metric("Incentivized Spam", f"{spam_pct:.1f}%")

            # Chart Distribution
            fig_pie = px.pie(
                df_batch,
                names="Verdict",
                title="Review Classification Distribution",
                color="Verdict",
                color_discrete_map={
                    "GENUINE": "#10B981",
                    "MACHINE_GENERATED_BOT": "#F59E0B",
                    "INCENTIVIZED_SPAM": "#EF4444"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(df_batch, use_container_width=True)

            csv_download = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated Batch Audit CSV",
                data=csv_download,
                file_name="review_radar_audit_results.csv",
                mime="text/csv"
            )

# ==========================================
# TAB 3: CLASSICAL NLP LABORATORY
# ==========================================
with tab3:
    st.subheader("🧪 Classical NLP Algorithm Laboratory")
    st.markdown("Test each syllabus-grounded component in isolation.")

    lab_choice = st.radio(
        "Choose Engine to Inspect:",
        options=["1. Text Normalizer", "2. Penn Treebank POS Profiler", "3. N-Gram Language Model & Smoothing"],
        horizontal=True
    )

    if "Normalizer" in lab_choice:
        st.markdown("#### Rule-Based Spelling & Orthographic Normalizer")
        lab_text = st.text_input("Input text with elongations, leetspeak, or punctuation abuse:", value="This is sooooo goooood!!!! b3st quality evvvver $$$$$")
        if st.button("Run Normalizer", key="btn_lab_norm"):
            norm_res = pipeline.normalizer.normalize(lab_text)
            st.json({
                "raw_text": norm_res.raw_text,
                "normalized_text": norm_res.normalized_text,
                "tokens": norm_res.tokens,
                "sentences": norm_res.sentences,
                "elongations_fixed": norm_res.elongations_fixed,
                "excessive_punct_fixed": norm_res.excessive_punct_fixed,
                "leetspeak_fixed": norm_res.leetspeak_fixed,
                "uppercase_ratio": norm_res.uppercase_ratio,
                "flags": norm_res.normalization_flags
            })

    elif "POS Profiler" in lab_choice:
        st.markdown("#### Penn Treebank POS Profiler & Syntactic Ratio Engine")
        lab_pos_text = st.text_input("Input text to extract POS distribution:", value="The compact thermal carafe brews exceptionally hot coffee.")
        if st.button("Run POS Profiler", key="btn_lab_pos"):
            toks = pipeline.normalizer.tokenize_words(lab_pos_text)
            pos_res = pipeline.pos_profiler.profile(toks)
            st.write("**Tagged Tokens:**", pos_res.tagged_tokens)
            st.json({
                "modifier_count_JJ_RB": pos_res.modifier_count,
                "content_count_NN_VB": pos_res.content_count,
                "modifier_to_content_ratio": pos_res.modifier_to_content_ratio,
                "superlative_density": pos_res.superlative_density,
                "type_token_ratio": pos_res.type_token_ratio,
                "flags": pos_res.anomaly_flags
            })

    else:
        st.markdown("#### N-Gram Language Model: Laplace vs. Good-Turing Comparison")
        lab_ngram_text = st.text_input("Input text to compute cross-entropy and perplexity:", value="The battery life exceeds my expectations.")
        if st.button("Run Perplexity Engine", key="btn_lab_lm"):
            sents = pipeline.normalizer.tokenize_sentences(lab_ngram_text)
            toks = pipeline.normalizer.tokenize_words(lab_ngram_text)
            pp_lap = pipeline.ngram_lm_laplace.evaluate_text(sents, toks, smoothing="laplace")
            pp_gt = pipeline.ngram_lm_good_turing.evaluate_text(sents, toks, smoothing="good_turing")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Laplace (Add-1) Results:**")
                st.json({
                    "perplexity": pp_lap.perplexity,
                    "log_likelihood": pp_lap.log_likelihood,
                    "sentence_perplexities": pp_lap.sentence_perplexities,
                    "flags": pp_lap.anomaly_flags
                })
            with c2:
                st.markdown("**Good-Turing (Katz) Results:**")
                st.json({
                    "perplexity": pp_gt.perplexity,
                    "log_likelihood": pp_gt.log_likelihood,
                    "sentence_perplexities": pp_gt.sentence_perplexities,
                    "flags": pp_gt.anomaly_flags
                })
