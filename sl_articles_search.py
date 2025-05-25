import streamlit as st
import subprocess
import pandas as pd
from biomcp_article_search import MedicalArticleSearch

# Set layout to wide
st.set_page_config(layout="wide")

def display_articles(articles):
    """Display articles in a formatted way"""
    if not articles:
        st.info("No valid articles found.")
        return

    # Display each article in a simple format
    for i, article in enumerate(articles, 1):
        st.markdown(f"""{i}. {article['title']}, {article['authors']}. {article['journal']}, {article['year']}. PMID: {article['pmid']}""", 
                        unsafe_allow_html=True)
            
def st_medial_articles_search():
    """Main function for the BioMCP Article Search Streamlit app"""
    st.title("🧬 BioMCP Article Search")

    disease = st.text_input("Enter disease name")

    if st.button("Search"):
       with st.spinner("🔍 Searching articles..."):
            search = MedicalArticleSearch()
            articles = search.search_articles(disease)
            display_articles(articles)


if __name__ == "__main__":
    st_medical_articles_search()
