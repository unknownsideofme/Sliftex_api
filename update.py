from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama.embeddings import OllamaEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from rapidfuzz import fuzz
from langchain.schema import Document
import json
import nltk
import pandas as pd
from metaphone import doublemetaphone

# pandas
data = pd.read_csv('./phonatics.csv')

# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index = "llama"
index = pc.Index(index)

#embed the query
embed_model = OllamaEmbeddings(model="llama3.2") 


# Initialize the BM25 encoder
encoder = BM25Encoder().default()

# Your data and BM25 encoding process
# Ensure the data column is converted to a list of strings

def add_title(title):
    corpus = [title.lower()]  # Convert the column to a list

    # Fit the encoder
    encoder.fit(corpus)
    encoder.dump("documentn.json")

    encoded_docs = BM25Encoder().load("document.json")

    retriever = PineconeHybridSearchRetriever(
        index=index,
        sparse_encoder=encoded_docs,
        embeddings=embed_model,
        top_k=30,
        alpha=0.5,
    )

    retriever.add_texts(corpus)

    # Generate metaphone values
    metA, metB = doublemetaphone(title)

    # Create a new row and append to the DataFrame
    new_row = {'title': title, 'metaphoneA': metA, 'metaphoneB': metB}
    global data  # Access the global DataFrame variable
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    # Update corpus and BM25 encoding
    corpus = data['title'].tolist()  # Convert the column to a list
    encoder.fit(corpus)
    encoder.dump("document.json")

    return "done"







    

    
    
    