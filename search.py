from flask import Flask, request, jsonify
import pandas as pd
import faiss
import numpy as np
import glob
import re
from openai import OpenAI
from google.cloud import storage
from io import BytesIO
import io
import os



app = Flask(__name__)


class SoundeffectRetriever():
    def __init__(self):
        self.bucket_name = "castpod-bucket"
        self.storage_client = storage.Client()
        self.openai_client = OpenAI(api_key="sk-proj-aQmCBiWvRBNHPc0UNxwmT3BlbkFJwc8m26f9Q8DQF9syEw9m")
        self.faiss_index = self.load_faiss_index_from_local()
        
    def load_url_df(self, local_file_path="/cache/soundeffects_urls.csv"):
        # Load the CSV data into a pandas DataFrame from local path
        df = pd.read_csv(local_file_path)
        return df
    
    def get_embedding(self, query, model="text-embedding-3-small"):
        return self.openai_client.embeddings.create(input=[query], model=model).data[0].embedding
    
    def load_faiss_index_from_local(self, local_file_path="/cache/faiss_index.index"):
        # Load the FAISS index from local path
        faiss_index = faiss.read_index(local_file_path)
        return faiss_index
    
    def search_faiss_index(self, query_embedding, k=1):
        distances, indices = self.faiss_index.search(np.array([query_embedding]).astype('float32'), k)
        url_df = self.load_url_df()
        url_list = url_df['download_link'].tolist()
        results = [url_list[idx] for i, idx in enumerate(indices[0])]
        return results

    def return_soundeffects(self, query):
        query_embedding = self.get_embedding(query)
        results = self.search_faiss_index(query_embedding)
        soundeffect_link = results[0]
        return soundeffect_link


# Initialize SoundeffectRetriever
retriever = SoundeffectRetriever()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        soundeffect_link = retriever.return_soundeffects(query)
        return jsonify({'url': soundeffect_link}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)