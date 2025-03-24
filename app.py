from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)  # Enable CORS to prevent cross-origin issues

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def scrape_website(url):
    """Scrape text from the provided URL with limits to prevent memory overload."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')

    # Limit the number of paragraphs extracted (adjust as needed)
    max_paragraphs = 50  
    text = " ".join([p.text for p in paragraphs[:max_paragraphs]])
    
    return text


def create_vector_store(text):
    """Convert text into embeddings and store in FAISS index."""
    chunks = text.split(". ")  # Split into sentences
    
    # Limit the number of chunks to avoid memory overload
    max_chunks = 500  # Adjust as needed
    chunks = chunks[:max_chunks]
    
    embeddings = model.encode(chunks, batch_size=8, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return index, chunks

def retrieve(query, index, chunks, top_k=3):
    """Retrieve relevant text chunks."""
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle function call from the agent and return the summary."""
    data = request.json
    print("Received Data:", data)  # Debugging print
    
    url = data.get("url")
    query = data.get("query")

    if not url or not query:
        return jsonify({"error": "Both 'url' and 'query' are required"}), 400

    # Scrape the website
    text = scrape_website(url)

    # Create FAISS index
    index, chunks = create_vector_store(text)

    # Retrieve relevant text
    retrieved_chunks = retrieve(query, index, chunks)

    # Return summarized content
    response = {"summary": " ".join(retrieved_chunks)}
    print("Response Sent:", response)  # Debugging print
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
