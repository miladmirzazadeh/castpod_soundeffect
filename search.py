from flask import Flask, request, jsonify, send_file
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
from google.cloud import storage
import io

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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



app = Flask(__name__)


class SoundeffectDownloader:
    def __init__(self):
        self.client_id = 'jCj2MBDQwUA5AmREUGxC'
        self.client_secret = 'AzIVXpziqensff2UI88xbJsw0An0x4683fcR7dke'
        self.username = 'milad1234'
        self.password = 'Milad_0816'
        self.authorization_url = 'https://freesound.org/apiv2/oauth2/authorize/?client_id=jCj2MBDQwUA5AmREUGxC&response_type=code&state=xyz'
        self.refresh_token = None
        self.access_token = None
        self.access_token = self.get_access_key()

    def init_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def get_access_key(self):
        if self.access_token:
            new_tokens = self.refresh_access_token()
            if new_tokens:
                self.access_token = new_tokens['access_token']
                self.refresh_token = new_tokens['refresh_token']
                return self.access_token
        else:
            authorization_code = self.get_authorization_code()
            if authorization_code:
                token_data = self.fetch_access_token(authorization_code)
                if token_data:
                    self.access_token = token_data['access_token']
                    self.refresh_token = token_data['refresh_token']
                    return self.access_token
        return None

    def fetch_access_token(self, authorization_code):
        try:
            token_url = 'https://freesound.org/apiv2/oauth2/access_token/'
            payload = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'authorization_code',
                'code': authorization_code
            }
            response = requests.post(token_url, data=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None

    def refresh_access_token(self):
        try:
            token_url = 'https://freesound.org/apiv2/oauth2/access_token/'
            payload = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token
            }
            response = requests.post(token_url, data=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None

    def get_authorization_code(self):
        self.driver = self.init_driver()
        try:
            self.driver.get(self.authorization_url)
            if self.check_element_by_xpath("//div[@class='container_main']//div[@class='container']//div[@style=\"font-size:14px;font-family:'Courier';\"]"):
                auth_code = self.extract_auth_code()
                if auth_code:
                    return auth_code
 
            if self.check_element_by_xpath("//input[@class='btn login large primary'][@name='allow']"):
                self.allow_authorization_button()
                return self.extract_auth_code()

            if self.check_element_by_xpath("//div[@class='row no-gutters']//form[@class='bw-form']//input[@id='id_username']"):
                self.submit_login_form()
                if self.check_element_by_xpath("//div[@class='container_main']//div[@class='container']//div[@style=\"font-size:14px;font-family:'Courier';\"]"):
                    auth_code = self.extract_auth_code()
                    if auth_code:
                        return auth_code
                else: 
                    self.allow_authorization_button()
                    return self.extract_auth_code()

        except Exception as e:
            return None
        finally:
            self.driver.quit()
        return None

    def check_element_by_xpath(self, xpath):
        try:
            self.driver.find_element(By.XPATH, xpath)
            return True
        except:
            return False

    def extract_auth_code(self):
        try:
            authorization_code = self.driver.find_element(By.XPATH, "//div[@class='container_main']//div[@class='container']//div[@style=\"font-size:14px;font-family:'Courier';\"]").text
            return authorization_code
        except:
            return None

    def allow_authorization_button(self):
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//input[@class='btn login large primary'][@name='allow']"))
            )
            authorize_button = self.driver.find_element(By.XPATH, "//input[@class='btn login large primary'][@name='allow']")
            if authorize_button.is_displayed() and authorize_button.is_enabled():
                authorize_button.click()
        except Exception as e:
            return None

    def submit_login_form(self):
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='row no-gutters']//form[@class='bw-form']//input[@id='id_username']"))
            )
            username_input = self.driver.find_element(By.XPATH, "//div[@class='row no-gutters']//form[@class='bw-form']//input[@id='id_username']")
            if username_input.is_displayed() and username_input.is_enabled():
                username_input.clear()
                username_input.send_keys(self.username)

            password_input = self.driver.find_element(By.XPATH, "//div[@class='row no-gutters']//form[@class='bw-form']//input[@id='id_password']")
            if password_input.is_displayed() and password_input.is_enabled():
                password_input.clear()
                password_input.send_keys(self.password)

            login_button = self.driver.find_element(By.XPATH, "//div[@class='row no-gutters']//form[@class='bw-form']//button[@type='submit']")
            if login_button.is_displayed() and login_button.is_enabled():
                login_button.click()
        except Exception as e:
            return None
            
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

    def return_soundeffect_id(self, query):
        query_embedding = self.get_embedding(query)
        results = self.search_faiss_index(query_embedding)
        soundeffect_id = results[0]
        return soundeffect_id


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
            else:
                logger.error(f"Unsupported audio format: {kind.extension}")
                return jsonify({"error": f"Unsupported audio format: {kind.extension}"}), 400
        
            return send_file(audio_buffer, as_attachment=True, download_name=f'new_soundeffect.{kind.extension}', mimetype=mimetype)

    except Exception as e:
        logger.error(f"Failed to send soundeffect to client: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)