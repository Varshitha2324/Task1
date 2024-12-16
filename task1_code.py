1. Install Required Libraries:
pip install PyMuPDF openai sentence-transformers faiss-cpu

2. Data Ingestion:
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)

    for page in document:
        text += page.get_text()
        
    document.close()
    return text

3. Chunk the Data:
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

4. Create Vector Embeddings:
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    return model.encode(chunks)

5. Store in a Vector Database:
import faiss
import numpy as np

def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create index
    index.add(np.array(embeddings).astype('float32'))  # Add embeddings to index
    return index
  
6. Query Handling:
def query_embedding(query):
    return model.encode([query])

Similarity Search:
def search(index, query_embeddings, k=5):
    distances, indices = index.search(np.array(query_embeddings).astype('float32'), k)
    return distances, indices

7. Generate Response:
def generate_response(relevant_chunks):
    # Here, you would synchronize with your LLM system (like OpenAI GPT)
    # For example:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": f"Please summarize these: {relevant_chunks}" }]
    # )
    # return response.choices[0].message['content']
    return "Generated response based on relevant chunks."


Example Use - Case:

# Main
pdf_path = "D:\ML/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf"  # Replace with the actual path
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = create_embeddings(chunks)
index = store_embeddings(embeddings)

query = "What is the unemployment rate based on the type of degree?"
query_vec = query_embedding(query)
_, indices = search(index, query_vec)

# Fetch relevant chunks for response generation
relevant_chunks = [chunks[i] for i in indices.flatten()]

final_response = generate_response(relevant_chunks)
print(final_response)

O/p:
Generated response based on relevant chunks.


