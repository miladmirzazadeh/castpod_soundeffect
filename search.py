# import pandas as pd
# import time
# import pandas as pd
# import faiss
# import numpy as np
# import glob
# import re
# from openai import OpenAI
# from dotenv import load_dotenv
# from google.cloud import storage
# from io import BytesIO
# import io
# import os


# class SoundeffectRetriever():
#     def __init__(self):
#         self.bucket_name = "castpod-bucket"
#         self.storage_client = storage.Client()
#         self.openai_client = client = OpenAI(api_key ="sk-proj-n23q1OUCkTQbXSUDI1uHT3BlbkFJ4oHjBodFeVpWA5cDDYGU")
#         self.faiss_index = self.download_faiss_index_to_memory()
        
#     def load_url_df(self, source_blob_name="soundeffects_urls.csv"):

#         # Get the bucket
#         bucket = self.storage_client.bucket(self.bucket_name)
        
#         # Get the blob
#         blob = bucket.blob(source_blob_name)
        
#         # Download the blob to an in-memory bytes buffer
#         csv_buffer = io.BytesIO()
#         blob.download_to_file(csv_buffer)
#         csv_buffer.seek(0)  # Reset buffer position to the beginning
        
#         # Load the CSV data into a pandas DataFrame
#         df = pd.read_csv(csv_buffer)
#         return df
    
#     def get_embedding(self, query,  model="text-embedding-3-small"):
#         return self.openai_client.embeddings.create(input = [query], model=model).data[0].embedding
    
#     def download_faiss_index_to_memory(self, source_blob_name="faiss_index.index"):
#         # Get the bucket
#         bucket = self.storage_client.bucket(self.bucket_name)
#         # Get the blob
#         blob = bucket.blob(source_blob_name) 
#         # Download the blob to an in-memory bytes buffer
#         index_buffer = io.BytesIO()
#         blob.download_to_file(index_buffer)
#         index_buffer.seek(0)  # Reset buffer position to the beginning   
#         # Load the FAISS index from the in-memory buffer
#         faiss_index = faiss.read_index(index_buffer)
#         return faiss_index


    
#     def search_faiss_index(self, query_embedding, k=1):
#         distances, indices = index.search(query_embedding, k)
#         url_df = self.load_url_df()
#         url_list = url_df['download_link'].tolist()
#         results = [url_list[idx] for i, idx in enumerate(indices[0])]
#         return results

#     def return_soundeffects(self, query):
#         query_embedding = self.get_embedding(query)
#         results = self.search_faiss_index(query_embedding)
#         soundeffect_link = results[0]
#         return soundeffect_link



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
        self.openai_client = OpenAI(api_key="sk-proj-PARlRihvR0E4jtCC2XAMT3BlbkFJuuGFu5dqYUTXZQ93sowT")
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