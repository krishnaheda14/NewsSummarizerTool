import os
import streamlit as st
import time
import numpy as np
import pandas as pd
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially Google API key)

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBhttY-1N20nwqjYS9GUVqxCCKX3wRzBQw"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
embeddings_file_path = "faiss_embeddings.npy"
metadata_file_path = "faiss_metadata.csv"

main_placeholder = st.empty()


class CustomVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)

    def save(self, embeddings_file_path, metadata_file_path):
        np.save(embeddings_file_path, self.embeddings)
        metadata = pd.DataFrame(
            [{"page_content": doc.page_content, "source": doc.metadata["source"]} for doc in self.documents])
        metadata.to_csv(metadata_file_path, index=False)

    @classmethod
    def load(cls, embeddings_file_path, metadata_file_path):
        if not os.path.exists(embeddings_file_path) or not os.path.exists(metadata_file_path):
            raise FileNotFoundError("Embeddings or metadata file not found.")

        embeddings = np.load(embeddings_file_path)
        metadata = pd.read_csv(metadata_file_path)
        if metadata.empty:
            raise ValueError("Metadata file is empty.")

        docs = [{"embedding": embedding, "metadata": meta} for embedding, meta in
                zip(embeddings, metadata.to_dict(orient='records'))]
        return cls(docs, embeddings)

    def query(self, query_embedding, top_k=3):
        dot_products = np.dot(self.embeddings, query_embedding)
        indices = np.argsort(dot_products)[::-1][:top_k]
        return [self.documents[i] for i in indices]


if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings using Gemini and save it to CustomVectorStore
    embeddings = []
    for doc in docs:
        embedding_result = genai.embed_content(
            model="models/embedding-001",
            content=doc.page_content,
            task_type="retrieval_document",
            title="Embedding of document"
        )
        embeddings.append(embedding_result["embedding"])

    vectorstore_gemini = CustomVectorStore(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the embeddings and metadata
    vectorstore_gemini.save(embeddings_file_path, metadata_file_path)

query = main_placeholder.text_input("Question: ")
if query:
    try:
        vectorstore = CustomVectorStore.load(embeddings_file_path, metadata_file_path)

        # Embed the query using Gemini
        query_embedding_result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_result["embedding"]

        # Find the closest documents
        sorted_docs = vectorstore.query(query_embedding, top_k=5)

        # Generate answer using the Gemini model
        combined_text = " ".join([doc["metadata"]["page_content"] for doc in sorted_docs])  # Combine top 3 docs
        generative_model = genai.GenerativeModel('gemini-pro')
        response = generative_model.generate_content(
            combined_text,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=1
            )
        )
        answer = response.text  # Access the text attribute directly

        st.header("Answer")
        st.write(answer)

        # Display sources
        st.subheader("Sources:")
        for doc in sorted_docs:
            st.write(doc["metadata"]["source"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
SEARCH_TERMS = [" "]
SEARCH_URL = " "
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

def search_arxiv_papers(terms):
    search_results = []

    for term in terms:
        query = term.replace(" ", "+")
        url = SEARCH_URL.format(query=query)

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract paper details
        entries = soup.find_all("li", class_="arxiv-result")
        for entry in entries:
            title = entry.find("p", class_="title").text.s