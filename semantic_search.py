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



# Load environment variables
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

index_semantic = "llama"
indexsemantic = pc.Index(index_semantic)

#embed the query
embed_model = OllamaEmbeddings(model="llama3.2") 




# Perform fuzzy matching



def fuzzy_match_titles(input_text, documents):
    """
    Function to perform fuzzy matching between the input text and a list of Document objects.
    It returns the documents sorted by their adjusted similarity score in descending order.

    Parameters:
    input_text (str): The text to match against the document titles.
    documents (list): A list of Document objects containing 'page_content' (titles).
    
    Returns:
    list: Sorted list of dictionaries with 'page_content' and 'adjusted_similarity_score'.
    """
    # Perform fuzzy matching and calculate similarity scores
    results = []
    for doc in documents:
        fuzzy_score = fuzz.ratio(input_text.lower(), doc.page_content.lower())
        original_score = doc.metadata.get('score', 0.0)
        
        # Calculate adjusted score based on fuzzy matching logic
        if fuzzy_score > 85:
            adjusted_score = fuzzy_score
        else:
            adjusted_score = (original_score*100 + fuzzy_score) / 2
        
        results.append({'page_content': doc.page_content, 'adjusted_similarity_score': adjusted_score})

    # Sort the results by adjusted similarity score in descending order
    sorted_results = sorted(results, key=lambda x: x['adjusted_similarity_score'], reverse=True)
    
    # Convert to JSON format
    formatted_response = {"similar titles": {}}
    for result in sorted_results:
        formatted_title = ' '.join([word.capitalize() for word in result['page_content'].split()])
        formatted_response["similar titles"][formatted_title] = {
            "score": result['adjusted_similarity_score']
        }

    return json.dumps(formatted_response, indent=4)






#creating pinecone retriever
encoded_docs = BM25Encoder().load("./document.json")
retriever = PineconeHybridSearchRetriever(index=indexsemantic, sparse_encoder=encoded_docs, embeddings=embed_model , top_k = 20 , alpha = 0.6)




# Initialize OpenAI model and embeddings (you can replace it with any other LLM you're using)

def semantic_search( title):
    # Generate the query embedding for the input title
    #title = clean_text(title)
    # Perform the semantic search in Pinecone using the embedding
    results = retriever.invoke(title.lower())
    print(results)
    results = fuzzy_match_titles(title, results)
    
    # Convert the Pinecone response to the desired JSON format
     
    
    return results


