from transformers import RobertaModel, RobertaTokenizer 
import torch
import logging 
import asyncio
import websockets
from datetime import datetime, timedelta
import json
import requests
import time
# from atproto import Client
import csv
import time
logging.getLogger("transformers").setLevel(logging.ERROR) # Roberta gives warning when not using pooler weights. We are not pooling, so currently disable warning. Revisit if wish to backprop to Roberta. 


"""
Should only be created once per run of main 
private data members: 
    __tokenizer
    __model
public data members: 
    tokens 
    cls_embedding
    text 
"""
class Roberta:
    output_dimension_size = 768 # static data member  
    def __init__(self):
        self.__tokenizer = None 
        self.__model = None
        self.text = None 
        self.tokens = None 
        self.cls_embedding = None 
        self.__init_model()
    def __init_model(self):
        model_name = "roberta-base"  
        self.__tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.__model = RobertaModel.from_pretrained(model_name) 
    def getTokenizer(self):
        return self.__tokenizer
    def getModel(self):
        return self.__model
    def getParameters(self):
        return self.__model.parameters()

    def setTextList(self, text): # text is str list 
        self.text = text
        self.__tokenizeText()
        self.__updateClsEmbedding()
    def getText(self):
        return self.text 
    def __tokenizeText(self):
        self.tokens=self.__tokenizer(self.text, return_tensors="pt", padding=True, truncation=True) # gotta check what the pt means; found in docs 
            
    
    def __updateClsEmbedding(self):
        if(self.tokens == None):
            raise Exception(f"Fatal error: attempt to update cls embedding without self.tokens set")
        outputs = self.__model(**self.tokens)
        self.cls_embedding = outputs.last_hidden_state[:, 0, :] # last hidden state 
    def getClsEmbedding(self):
        return self.cls_embedding

class BlueskyGetter:
    def __init__(self):
        self.roberta = Roberta()
        
        # 1. Create a persistent session to keep TCP connections alive
        self.session = requests.Session()
        self.headers = None
        
        # 2. Authenticate once upon creation
        self._authenticate()

    def _authenticate(self):
        """Internal method to handle Bluesky login."""
        login_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
        credentials = {
            "identifier": "stars-and-dashes.bsky.social",
            "password": "tn7w-aws5-4axi-hoak" 
        }

        print("Authenticating with Bluesky...")
        # Use the session for the login request
        response = self.session.post(login_url, json=credentials)

        if response.status_code != 200:
            raise Exception(f"Fatal Login Error: {response.json()}")

        access_token = response.json().get('accessJwt')
        
        # Store the headers globally for the instance
        self.headers = {"Authorization": f"Bearer {access_token}"}
        
        # We can also attach the headers directly to the session so we don't 
        # have to pass them manually in every GET request!
        self.session.headers.update(self.headers)
        print("Logged in successfully.")

    def fetch_CLS(self, date, ticker, limit=5):
        """
        Fetches posts and returns the average CLS embedding.
        """
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            next_day = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            print("Error: Date must be in 'YYYY-MM-DD' format.")
            return None

        # Search for Historical Posts
        search_url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
        advanced_query = f"{ticker} since:{date} until:{next_day}"
        
        valid_texts = []
        cursor = None

        while len(valid_texts) < limit:
            params = {
                'q': advanced_query,  
                'limit': 100 
            }
            if cursor:
                params['cursor'] = cursor

            # Use self.session.get() instead of requests.get()
            # We also don't need to pass headers here anymore since they are on the session
            response = self.session.get(search_url, params=params)

            # --- Important Edge Case Handling ---
            # API tokens expire (usually after 2 hours). If your neural net takes a long 
            # time to run, you might get a 401 Unauthorized. This catches that and logs back in.
            if response.status_code == 401:
                print("Token expired. Re-authenticating...")
                self._authenticate()
                continue
            
            # ─── NEW: RATE LIMIT HANDLER ────────────────────────────
            if response.status_code == 429:
                print("\n[429 Rate Limit] Bluesky server timeout. Sleeping for 60 seconds...")
                time.sleep(60)
                continue # Do not break; try the exact same request again
            # ────────────────────────────────────────────────────────

            if response.status_code != 200:
                print(f"Search failed: {response.status_code} - {response.text}")
                break

            data = response.json()
            posts = data.get('posts', [])

            if not posts:
                break 

            for post in posts:
                if post['record']['createdAt'].startswith(date):
                    text = post['record']['text'].replace('\n', ' ')
                    valid_texts.append(text)
                    
                    if len(valid_texts) == limit:
                        break 

            cursor = data.get('cursor')
            if not cursor or len(valid_texts) >= limit:
                break

            time.sleep(0.5) 

        if not valid_texts:
            print(f"No posts found for {ticker} on {date}.")
            return None

        # Generate and Average Embeddings
        self.roberta.setTextList(valid_texts)
        embeddings = self.roberta.getClsEmbedding()
        avg_embedding = torch.mean(embeddings, dim=0)
        
        return avg_embedding
