from flask import Flask, request, jsonify, send_file
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery

import io

from openai import OpenAI
import filetype



# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

import requests
from io import BytesIO

import logging
from google.cloud import logging as cloud_logging






# Set up Google Cloud Logging
client = cloud_logging.Client()
client.setup_logging()

# Set up standard Python logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

openai_client = OpenAI(api_key = "sk-accounttt-jGAyAXNihT8N5Gq0HvsHT3BlbkFJqGP18q6BNJOT37ozTQeG")
bigquery_client = bigquery.Client(project = "castpodproject")


app = Flask(__name__)


class SoundeffectDownloader:
    def __init__(self):
        self.client_id = 'jCj2MBDQwUA5AmREUGxC'
        self.client_secret = 'AzIVXpziqensff2UI88xbJsw0An0x4683fcR7dke'
        self.username = 'milad1234'
        self.password = 'Milad_0816'
        self.refresh_token = self.get_secret_key("freesound_refresh_token")
        self.access_token = self.get_secret_key("freesound_access_token")

    def set_secret_key(self, secret_id, secret_value):
        # Create the Secret Manager client
        secret_client = secretmanager.SecretManagerServiceClient()
        # Define the resource name of the secret
        project_id = "castpodproject"
        parent = f"projects/{project_id}/secrets/{secret_id}"
        # Add the secret value as a new version
        payload = secret_value.encode("UTF-8")
        response = secret_client.add_secret_version(
            request={"parent": parent, "payload": {"data": payload}}
        )
        return response.name

    def get_secret_key(self, secret_id):
        # Create the Secret Manager client
        secret_client = secretmanager.SecretManagerServiceClient()
        # Define the resource name of the secret
        project_id = "castpodproject"
        secret_id = secret_id
        version_id = "latest"  # or specify a version number if needed
        # Build the resource name
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        # Access the secret version
        response = secret_client.access_secret_version(name=name)
        # Get the secret payload and decode it
        secret_payload = response.payload.data.decode("UTF-8")
        return secret_payload


    def get_access_key(self):
        if self.access_token:
            new_tokens = self.refresh_access_token()
            if new_tokens:
                return self.access_token
        return None


    def refresh_access_token(self):
        # try:
        token_url = 'https://freesound.org/apiv2/oauth2/access_token/'
        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token
        }
        response = requests.post(token_url, data=payload)
        if response.status_code == 200:
            new_tokens = response.json()
            self.access_token = new_tokens['access_token']
            self.set_secret_key("freesound_access_token", self.access_token)
            self.refresh_token = new_tokens['refresh_token']
            self.set_secret_key("freesound_refresh_token", self.refresh_token)
            logger.info(f"new tokens : {new_tokens}")
            return response.json()
        else:
            logger.info("response: {}".format(response.json()))

        # except Exception as e:
        #     logger.error(f"Failed to refresh freesound token: {e}", exc_info=True)
        #     return None

   
    def download_soundeffect(self, sound_id):
        try: 
            url = f'https://freesound.org/apiv2/sounds/{sound_id}/download/'
            # Set the headers
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            # Make the GET request
            response = requests.get(url, headers=headers)
        
            # Check if the request was successful
            if response.status_code == 200:
                wav_buffer = BytesIO(response.content)
                wav_buffer.seek(0)  # Move the cursor to the beginning of the buffer
                return wav_buffer
            elif response.status_code != 200: 
                logger.info("tyring to refresh token")
                self.refresh_access_token()
                url = f'https://freesound.org/apiv2/sounds/{sound_id}/download/'
                # Set the headers
                headers = {
                    'Authorization': f'Bearer {self.access_token}'
                }
                # Make the GET request
                response = requests.get(url, headers=headers)
            
                # Check if the request was successful
                if response.status_code == 200:
                    wav_buffer = BytesIO(response.content)
                    wav_buffer.seek(0)  # Move the cursor to the beginning of the buffer
                    return wav_buffer

        except Exception as e:
            logger.error(f"Failed to download soundeffect: {e}", exc_info=True)
            return False


class SoundeffectRetriever():
    def __init__(self):
        self.bucket_name = "castpod-bucket"
        self.storage_client = storage.Client()
        self.openai_client = OpenAI(api_key="sk-myserviceaccount-ktRYtzqIswjehExqOUaxT3BlbkFJXzs3otofvW6Rx1nqEc0e")
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
        id_list = url_df['id'].tolist()
        results = [id_list[idx] for i, idx in enumerate(indices[0])]
        return results

    def select_soundeffect(self, desc, soundeffects):
        select_se_prompt = '''
        You are an expert soundeffect selector. Based on a description the narrator gives you, your task is to select a soundeffect from a list of options to be placed within a podcast episode. The options hvae different descriptions and different durations. 
        Your output is only an id of the best choice: for example, correct: "112223" , wrong: "id: 112223" since I will convert your output to int directly. We will afterward, trim the first 5 seconds(or less) of the selected sound effect, so too large or too short sound files may not be good choice for us. 
        However, the descriptions (including tags, and captions) are more important. Select something that you predict is more useful in the context of the podcast and as soundeffects.
        The desired narrator description is: {}, 
        list of options: 
        {}
        '''
        response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": select_se_prompt.format(desc, soundeffects)
            }
        ],
        temperature=0.01,
        max_tokens=20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        choice = response.choices[0].message.content
        return int(choice)
    def get_soundeffects_desc(self, se_ids):
        ids_to_query = se_ids
        # Define the query
        query = """
            SELECT *
            FROM `castpodproject.freesound_soundeffects.id_duration_desc`
            WHERE id IN UNNEST(@ids)
        """


        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "INT64", ids_to_query)
            ]
        )

        # Run the query
        query_job = bigquery_client.query(query, job_config=job_config)

        options_desciptions = ""
        # Fetch and print results
        rows = query_job.result()
        for row in rows:
            options_desciptions+= f"ID: {row.id} => Description: {row.complete_desc} , /n Duration => {row.duration} /n "
        return options_desciptions

    def return_soundeffect_id(self, query, k=5):
        query_embedding = self.get_embedding(query)
        soundeffect_ids = self.search_faiss_index(query_embedding, k)
        #retrieve descriptions from bigquery
        descriptions = self.get_soundeffects_desc(soundeffect_ids)
        #using GPT
        final_soundeffect_id = self.select_soundeffect(desc=query, soundeffects= descriptions)
        return final_soundeffect_id




# Initialize SoundeffectRetriever
retriever = SoundeffectRetriever()
soundeffect_downloader = SoundeffectDownloader()


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    if not query:
        logger.error(f'error : No query provided')
        return jsonify({'error': 'No query provided'}), 400
    try:
        soundeffect_id = retriever.return_soundeffect_id(query)
        audio_buffer = soundeffect_downloader.download_soundeffect(soundeffect_id)
        if audio_buffer:
            # Detect the file type
            kind = filetype.guess(audio_buffer)
            if kind is None:
                logger.error("Cannot guess the file type!")
                return jsonify({"error": "Cannot guess the file type"}), 400
            
            # Set the correct mimetype based on file type
            if kind.extension == "wav":
                mimetype = 'audio/wav'
            elif kind.extension == "mp3":
                mimetype = 'audio/mpeg'
            elif kind.extension == "ogg":
                mimetype = 'audio/ogg'
            elif kind.extension == "aiff":
                mimetype = 'audio/aiff'
            elif kind.extension == "flac":
                mimetype = 'audio/flac'
            else:
                logger.error(f"Unsupported audio format: {kind.extension}")
                return jsonify({"error": f"Unsupported audio format: {kind.extension}"}), 400

            
            return send_file(audio_buffer, as_attachment=True, download_name=f'new_soundeffect.{kind.extension}', mimetype=mimetype)

    except Exception as e:
        logger.error(f"Failed to send soundeffect to client: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500




@app.route('/only_search', methods=['POST'])
def only_search():
    data = request.json
    query = data.get('query')
    if not query:
        logger.error(f'error : No query provided')
        return jsonify({'error': 'No query provided'}), 400
    try:
        soundeffect_ids = retriever.return_soundeffect_id(query, 5)
        return jsonify({"ids": soundeffect_ids}), 200
    except Exception as e:
        logger.error(f"Failed to search soundeffects {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
@app.route('/only_download', methods=['POST'])
def only_download():
    data = request.json
    soundeffect_id = data.get('soundeffect_id')
    try:
        audio_buffer = soundeffect_downloader.download_soundeffect(soundeffect_id)
        logger.info(f"audio_buffer : {audio_buffer}")
        if audio_buffer:
            # Detect the file type
            kind = filetype.guess(audio_buffer)
            if kind is None:
                logger.error("Cannot guess the file type!")
                return jsonify({"error": "Cannot guess the file type"}), 400
            
            # Set the correct mimetype based on file type
            if kind.extension == "wav":
                mimetype = 'audio/wav'
            elif kind.extension == "mp3":
                mimetype = 'audio/mpeg'
            elif kind.extension == "ogg":
                mimetype = 'audio/ogg'
            elif kind.extension == "aiff":
                mimetype = 'audio/aiff'
            elif kind.extension == "flac":
                mimetype = 'audio/flac'
            else:
                logger.error(f"Unsupported audio format: {kind.extension}")
                return jsonify({"error": f"Unsupported audio format: {kind.extension}"}), 400
        
            return send_file(audio_buffer, as_attachment=True, download_name=f'new_soundeffect.{kind.extension}', mimetype=mimetype)

    except Exception as e:
        logger.error(f"Failed to send soundeffect to client: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)