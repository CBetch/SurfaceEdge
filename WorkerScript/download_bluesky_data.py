import asyncio
import websockets
import json
import requests
import time
# from atproto import Client
import csv
import time

def fetch_history_to_csv():
    # 1. Manually Log In (Create Session)
    login_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    credentials = {
        "identifier": "stars-and-dashes.bsky.social", 
        "password": "tn7w-aws5-4axi-hoak"
    }
    
    print("Logging in...")
    login_response = requests.post(login_url, json=credentials)
    
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.json()}")
        return
        
    # Extract the secure token to use for our searches
    access_token = login_response.json().get('accessJwt')
    headers = {"Authorization": f"Bearer {access_token}"}
    print("Logged in successfully!\n")

    # 2. Search for Historical Posts
    search_url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    target_amount = 250
    all_posts = []
    cursor = None

    print(f"Fetching {target_amount} posts...")
    
    while len(all_posts) < target_amount:
        params = {
            'q': 'MSFT',  # Your search keyword
            'limit': 100
        }
        if cursor:
            params['cursor'] = cursor

        # Ask for the data using our secure token
        response = requests.get(search_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Search failed: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        posts = data.get('posts', [])
        
        if not posts:
            break
            
        # Extract and clean the data
        for post in posts:
            all_posts.append({
                'date': post['record']['createdAt'],
                'author': post['author']['handle'],
                'text': post['record']['text'].replace('\n', ' ')
            })
            
            if len(all_posts) >= target_amount:
                break
                
        # Grab the cursor for the next page
        cursor = data.get('cursor')
        if not cursor:
            break
            
        print(f"Collected {len(all_posts)} posts so far...")
        time.sleep(1) # Be polite to the API

    # 3. Save to CSV
    csv_filename = 'bluesky_dataset.csv'
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'author', 'text'])
        writer.writeheader()
        writer.writerows(all_posts)

    print("-" * 50)
    print(f"Success! Saved {len(all_posts)} posts to {csv_filename}")


if __name__ == "__main__":
    fetch_history_to_csv()
"""def fetch_public_history():
    # The public, unauthenticated Bluesky API endpoint
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    
    all_posts = []
    cursor = None
    target_amount = 250  # How many posts you want for your dataset
    
    print(f"Fetching {target_amount} posts without a password...\n")
    print("-" * 50)
    
    while len(all_posts) < target_amount:
        # 1. Set up our search parameters
        params = {
            'q': 'machine learning',  # Try changing this to your topic!
            'limit': 100,             # 100 is the max the API allows per request
            'sort': 'latest'
        }
        
        # If we have a cursor from a previous loop, tell the API where we left off
        if cursor:
            params['cursor'] = cursor
            
        # 2. Ask the API for the data
        response = requests.get(url, params=params)
        
        # Stop if something goes wrong (like a server error)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
            
        data = response.json()
        posts = data.get('posts', [])
        
        # Stop if there are no more posts to find
        if not posts:
            break
            
        # 3. Extract the text and author from the JSON
        for post in posts:
            author = post['author']['handle']
            text = post['record']['text']
            date = post['record']['createdAt']
            
            all_posts.append({'author': author, 'text': text})
            
            # Print a little preview to the console so we know it's working
            # We replace newlines with spaces so it prints neatly on one line
            clean_text = text.replace('\n', ' ')
            print(f"[{date[:10]}] @{author}: {clean_text[:60]}...")
            
            # Stop if we hit our exact target number
            if len(all_posts) >= target_amount:
                break
                
        # 4. Grab the cursor for the next page of results
        cursor = data.get('cursor')
        if not cursor:
            break # No more pages exist
            
        # Be polite to the free API so we don't get blocked
        time.sleep(1)
        
    print("-" * 50)
    print(f"Success! Collected {len(all_posts)} total posts.")

if __name__ == "__main__":
    fetch_public_history()
# The Bluesky Jetstream endpoint. 
# wantedCollections = feed.post because we want to filter out likes, follows, and profile updates 
JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

# async for endpoint interaction 
async def listen_to_firehose():
    print("Connecting to Bluesky Jetstream...")
    
    async with websockets.connect(JETSTREAM_URL) as websocket:
        print("Connected! Listening for live posts... (Press Ctrl+C to stop)\n")
        print("-" * 50)
        
        try:
            while True:
                # await the raw message from the server
                message = await websocket.recv()
                
                # Parse the JSON data
                data = json.loads(message)
                
                # find the actual post content
                # Jetstream sends events; we only care about 'commit' events (new/updated data)
                # that have a 'record' (the actual post payload).
                if data.get('commit') and data['commit'].get('record'):
                    record = data['commit']['record']
                    
                    # Extract the text and language tags
                    text = record.get('text', '')
                    langs = record.get('langs', ['unknown'])
                    
                    # Filter for English posts
                    if text and 'en' in langs:
                        print(f"[EN] {text}")
                        print("-" * 50)
                        
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed by the server.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    try:
        # Run the asynchronous event loop
        asyncio.run(listen_to_firehose())
    except KeyboardInterrupt:
        print("\nDisconnected from the firehose.")
"""
