import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from vector import retriever, df  

st.set_page_config(page_title="Pizza Restaurant AI & Analytics", layout="wide")
st.title("🍕 Pizza Restaurant Review Analyzer & Insights")

# ==========================================
# DATA SCIENCE PROCESSING BACKGROUND LAYERS
# ==========================================
@st.cache_data
def run_sentiment_analysis(_data_frame):
    processed_df = _data_frame.copy()
    def get_sentiment(text):
        score = TextBlob(str(text)).sentiment.polarity
        if score > 0.1: return "Positive"
        elif score < -0.1: return "Negative"
        return "Neutral"
    processed_df['Computed_Sentiment'] = processed_df['Review'].apply(get_sentiment)
    return processed_df

analyzed_df = run_sentiment_analysis(df)

@st.cache_data
def extract_lda_topics(text_series, num_topics=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(text_series.astype(str))
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    words = vectorizer.get_feature_names_out()
    topics_list = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-5:-1]]
        topics_list.append(f"Theme {topic_idx+1}: " + ", ".join(top_words))
    return topics_list

discovered_topics = extract_lda_topics(analyzed_df['Review'])

# ==========================================
# STREAMLIT USER INTERFACE TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🤖 AI Query Assistant", "📈 EDA & Sentiment Trends", "🏷️ Topic Modeling (LDA)"])

with tab1:
    st.header("Ask Questions to the AI Expert")
    question = st.text_input("What would you like to know about customer experiences?")
    
    if question:
        with st.spinner("Searching vectors locally..."):
            # Fetch matching documents from your local server storage
            retrieved_reviews = retriever.invoke(question)
            
            # Combine the relevant items to create an analytical summary
            st.markdown("### 📝 Verified Review Context Summary:")
            
            summary_box = "Here are the top matches found in your data regarding your query:\n\n"
            for doc in retrieved_reviews:
                summary_box += f"• *[{doc.metadata.get('rating')} Stars]*: {doc.page_content}\n\n"
            
            st.info(summary_box)

with tab2:
    st.header("Data Distribution & Sentiment Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews Count", len(analyzed_df))
    col2.metric("Positive Comments", len(analyzed_df[analyzed_df['Computed_Sentiment'] == 'Positive']))
    col3.metric("Negative Comments", len(analyzed_df[analyzed_df['Computed_Sentiment'] == 'Negative']))
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_sent = px.histogram(analyzed_df, x="Computed_Sentiment", color="Computed_Sentiment",
                                title="Calculated Sentiment Proportions",
                                color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"})
        st.plotly_chart(fig_sent, use_container_width=True)
    with col_chart2:
        fig_rate = px.box(analyzed_df, y="Rating", title="Customer Star Rating Distributions")
        st.plotly_chart(fig_rate, use_container_width=True)

with tab3:
    st.header("Algorithmic Key Topic Mining")
    for topic in discovered_topics:
        st.success(topic)
