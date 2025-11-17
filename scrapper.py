pip install requests beautifulsoup4 pandas nltk
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

class EthicalScraper:
    def __init__(self, base_url, limit=100):
        self.base_url = base_url
        self.limit = limit
        self.data = []
        # Ethical Header: Identify yourself
        self.headers = {
            'User-Agent': 'StudentNLPProject/1.0 (contact: student@example.com)'
        }

    def fetch_page(self, url):
        """Fetches the HTML content of a page."""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Failed to retrieve {url}: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_html(self, html):
        """Parses HTML to extract Title, Content, and Metadata."""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []

        # Note: These selectors are specific to books.toscrape.com
        # For a real news site, you would change 'article' to the specific div class
        items = soup.find_all('article', class_='product_pod')

        for item in items:
            try:
                # Extract Title
                title_tag = item.find('h3').find('a')
                title = title_tag['title']
                
                # Extract 'Content' (In this sandbox, price/stock acts as content metadata)
                # On a news site, this would be: item.find('div', class_='article-body').text
                price = item.find('p', class_='price_color').text
                availability = item.find('p', class_='instock availability').text.strip()
                
                # Combine for a "text document" simulation
                full_text = f"{title}. Price: {price}. Status: {availability}"

                articles.append({
                    'title': title,
                    'content': full_text, # Raw text
                    'source_url': self.base_url
                })
            except AttributeError:
                continue
        
        return articles

    def run(self):
        print(f"Starting scrape of {self.base_url}...")
        
        # Logic to paginate (scrape multiple pages to get 100+ docs)
        page_num = 1
        while len(self.data) < self.limit:
            target_url = f"{self.base_url}catalogue/page-{page_num}.html"
            print(f"Scraping: {target_url}")
            
            html = self.fetch_page(target_url)
            if not html:
                break # Stop if page doesn't exist
            
            batch = self.parse_html(html)
            if not batch:
                break
                
            self.data.extend(batch)
            page_num += 1
            
            # ETHICAL STEP: Rate Limiting
            sleep_time = random.uniform(1, 3) # Sleep between 1 and 3 seconds
            time.sleep(sleep_time)

        # Truncate to limit and save
        final_data = self.data[:self.limit]
        df = pd.DataFrame(final_data)
        df.to_csv('scraped_dataset.csv', index=False)
        print(f"\nSuccess! Collected {len(final_data)} documents. Saved to 'scraped_dataset.csv'.")
        return df

# --- EXECUTION ---
# We use a safe sandbox URL. 
# If scraping a real news site, replace with specific URL and update parse_html selectors.
base_url = "http://books.toscrape.com/"
scraper = EthicalScraper(base_url, limit=100)
raw_df = scraper.run()

import nltk
import re

# Download necessary NLTK data (only need to run once)
nltk.download('punkt')
nltk.download('punkt_tab')

def nlp_pipeline(text):
    # --- STEP 3: Text Cleaning ---
    
    # 1. Remove HTML tags (Regex)
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # 3. Remove special characters (keep only letters and numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # --- STEP 4: Pre-processing ---
    
    # 4. Lowercasing
    text = text.lower()
    
    # 5. Tokenization
    tokens = nltk.word_tokenize(text)
    
    return tokens

# Apply the pipeline to our scraped DataFrame
# We check if data exists first
if 'raw_df' in locals():
    # Create a new column for tokens
    raw_df['cleaned_tokens'] = raw_df['content'].apply(nlp_pipeline)
    
    # Join tokens back to string for easy reading in preview
    raw_df['cleaned_text'] = raw_df['cleaned_tokens'].apply(lambda x: ' '.join(x))
    
    print(raw_df[['content', 'cleaned_tokens']].head())
from collections import Counter
import numpy as np

def analyze_language(df):
    all_tokens = [word for tokens in df['cleaned_tokens'] for word in tokens]
    
    # 1. Vocabulary Size (Unique words)
    vocab_size = len(set(all_tokens))
    
    # 2. Total Word Count
    total_words = len(all_tokens)
    
    # 3. Average Sentence/Document Length
    doc_lengths = df['cleaned_tokens'].apply(len)
    avg_doc_length = np.mean(doc_lengths)
    
    # 4. Most Common Words (Frequency Distribution)
    common_words = Counter(all_tokens).most_common(10)
    
    print("\n--- Language Analysis Statistics ---")
    print(f"Total Documents Analyzed: {len(df)}")
    print(f"Total Word Count: {total_words}")
    print(f"Vocabulary Size (Unique Words): {vocab_size}")
    print(f"Average Words per Document: {avg_doc_length:.2f}")
    print("\nTop 10 Most Common Words:")
    for word, freq in common_words:
        print(f"{word}: {freq}")

# Run analysis
if 'raw_df' in locals():
    analyze_language(raw_df)

