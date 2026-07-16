import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, df  

st.set_page_config(page_title="Pizza Restaurant AI & Analytics", layout="wide")
st.title("🍕 Pizza Restaurant Review Analyzer & Insights")

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

#Sentiment Scoring Engine
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

#LDA Topic Modeling Engine
@st.cache_data
def extract_lda_topics(text_series, num_topics=3):
    # Vectorize and strip out common english words
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(text_series.astype(str))
    
    # Mathematical cluster fitting
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

# TAB 1: LangChain Ollama RAG Assistant
with tab1:
    st.header("Ask Questions to the AI Expert")
    question = st.text_input("What would you like to know about customer experiences? (e.g., 'Is the cheese crust liked?')")
    
    if question:
        with st.spinner("Searching vectors and running Llama 3.2..."):
            # Execute your exact backend LangChain logic pipeline
            retrieved_reviews = retriever.invoke(question)
            result = chain.invoke({"reviews": retrieved_reviews, "question": question})
            
            # Render output nicely on the screen
            st.markdown("### 📝 AI Answer:")
            st.info(result)
            
        # UI expansion to see referenced source vectors
        with st.expander("🔍 View Raw Retrieved Context Reviews"):
            for i, doc in enumerate(retrieved_reviews):
                st.write(f"**Doc {i+1}** (Rating: {doc.metadata.get('rating')}): {doc.page_content}")

# TAB 2: Exploratory Data Analysis & Graphing
with tab2:
    st.header("Data Distribution & Sentiment Analytics")
    
    # Top KPI Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews Count", len(analyzed_df))
    col2.metric("Positive Comments", len(analyzed_df[analyzed_df['Computed_Sentiment'] == 'Positive']))
    col3.metric("Negative Comments", len(analyzed_df[analyzed_df['Computed_Sentiment'] == 'Negative']))
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Sentiment Chart
        fig_sent = px.histogram(analyzed_df, x="Computed_Sentiment", color="Computed_Sentiment",
                                title="Calculated Sentiment Proportions",
                                color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"})
        st.plotly_chart(fig_sent, use_container_width=True)
        
    with col_chart2:
        # Rating Distribution Chart from your CSV Metadata
        fig_rate = px.box(analyzed_df, y="Rating", title="Customer Star Rating Distributions")
        st.plotly_chart(fig_rate, use_container_width=True)

# TAB 3: Latent Dirichlet Allocation Insights
with tab3:
    st.header("Algorithmic Key Topic Mining")
    st.write("The scikit-learn Latent Dirichlet Allocation (LDA) algorithm mined these recurring semantic keywords from the data:")
    
    for topic in discovered_topics:
        st.success(topic)
        
    st.markdown("### How this algorithm helps:")
    st.caption("Unlike an LLM which generates conversational text, LDA mathematically assesses word co-occurrences. If words like 'crust, cheese, topping' consistently appear close together across rows, the model isolates them as an independent consumer trend cluster.")
