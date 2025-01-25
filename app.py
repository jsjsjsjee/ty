from flask import Flask, render_template, request, redirect, url_for
import os
import json
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = Flask(__name__)

# Configurations
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"txt", "pdf", "png", "jpg", "jpeg", "mp4"}
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Persistent storage files
FEEDBACK_FILE = "feedback.json"
UPLOADS_FILE = "uploads.json"

# Initialize files if not exist
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(UPLOADS_FILE):
    with open(UPLOADS_FILE, "w") as f:
        json.dump([], f)

# Load initial data from files
with open(FEEDBACK_FILE, "r") as f:
    feedback_store = json.load(f)

with open(UPLOADS_FILE, "r") as f:
    uploads_store = json.load(f)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    print(f"NLTK data download failed: {e}")

class SimpleNewsAnalyzer:
    def __init__(self):
        self.suspicious_words = {
            'shocking', 'unbelievable', 'miracle', 'secret', 'conspiracy',
            'hoax', 'scandal', 'anonymous', 'claims', 'allegedly', 'rumor',
            'sources say', 'viral', 'sensational', 'you won\'t believe'
        }
        self.credible_words = {
            'research', 'study', 'evidence', 'expert', 'official', 
            'according to', 'report', 'analysis', 'investigation',
            'confirmed', 'verified', 'sources confirmed'
        }
        
    def analyze_text(self, text):
        text = text.lower()
        suspicious_count = sum(1 for word in self.suspicious_words if word in text)
        credible_count = sum(1 for word in self.credible_words if word in text)
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        has_sources = len(urls) > 0
        
        if suspicious_count > credible_count + 1:
            return {"labels": ["potentially misleading"], "scores": [0.7]}
        elif credible_count > suspicious_count or has_sources:
            return {"labels": ["likely reliable"], "scores": [0.7]}
        else:
            return {"labels": ["unverified"], "scores": [0.5]}

fake_news_detector = SimpleNewsAnalyzer()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def fetch_news(query):
    """Fetch news using a public API like NewsAPI."""
    try:
        api_key = "449b98e7d52e49ee901f800a11ad5db1"  # Ensure this key is valid
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&apiKey={api_key}&pageSize=10&language=en&sortBy=relevancy"
        )
        print(f"Making request to NewsAPI with query: {query}")
        
        response = requests.get(url)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"NewsAPI error: {response.json().get('message', 'Unknown error')}"
            print(error_msg)
            raise Exception(error_msg)
        
        news_data = response.json()
        articles = news_data.get("articles", [])
        print(f"Found {len(articles)} articles")
        
        valid_articles = []
        for article in articles:
            if article.get("title") and article.get("description") and article.get("url"):
                cleaned_article = {
                    "title": article["title"].strip(),
                    "description": article["description"].strip(),
                    "url": article["url"],
                    "urlToImage": article.get("urlToImage", ""),
                    "source": article.get("source", {}).get("name", "Unknown Source")
                }
                valid_articles.append(cleaned_article)
        
        print(f"Returning {len(valid_articles)} valid articles")
        return valid_articles
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
        raise Exception("Failed to connect to news service. Please check your internet connection.")
    except Exception as e:
        print(f"Error in fetch_news: {str(e)}")
        raise

def save_feedback():
    """Save feedback data to the feedback JSON file."""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_store, f)

def save_uploads():
    """Save uploads data to the uploads JSON file."""
    with open(UPLOADS_FILE, "w") as f:
        json.dump(uploads_store, f)

def get_feedback_percentages(news_id):
    """Calculate real and fake feedback percentages for a given news ID."""
    feedback = feedback_store.get(news_id, {"real": 0, "fake": 0})
    total = feedback["real"] + feedback["fake"]
    if total > 0:
        real_percentage = (feedback["real"] / total) * 100
        fake_percentage = (feedback["fake"] / total) * 100
    else:
        real_percentage = fake_percentage = 0
    return {"real_percentage": round(real_percentage, 2), "fake_percentage": round(fake_percentage, 2)}

def is_valid_url(url):
    """Check if the input is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_url_content(url):
    """Fetch and extract text content from a URL."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'title': soup.title.string if soup.title else url,
            'description': text[:500],  # First 500 characters as description
            'url': url
        }
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    articles = []
    related_uploads = []
    error_message = None
    search_performed = False  # Add this flag
    
    if request.method == "POST":
        try:
            query = request.form.get("query", "").strip()
            if not query:
                error_message = "Please enter a search query"
                print(error_message)  # Debugging print
                return render_template("index.html", articles=[], uploads=[], error_message=error_message, search_performed=False)
            
            print(f"Received search query: {query}")
            search_performed = True  # Set the flag when search is performed
            
            # Check if the query is a URL
            if is_valid_url(query):
                print("Processing as URL")
                url_content = fetch_url_content(query)
                if url_content:
                    articles = [url_content]
                else:
                    error_message = "Could not fetch content from the provided URL"
                    print(error_message)  # Debugging print
            else:
                print("Processing as news search")
                articles = fetch_news(query)

            print(f"Retrieved {len(articles)} articles")  # Log article count

            # Analyze each article
            for index, article in enumerate(articles):
                try:
                    text_to_analyze = f"{article['title']} {article.get('description', '')}"
                    prediction = fake_news_detector.analyze_text(text_to_analyze)
                    article["prediction"] = prediction["labels"][0]
                    article["confidence"] = f"{prediction['scores'][0] * 100:.0f}%"
                except Exception as e:
                    print(f"Analysis error for article {index}: {str(e)}")
                    article["prediction"] = "unable to analyze"
                    article["confidence"] = "N/A"
                
                article["id"] = str(index)

                # Calculate feedback percentages
                feedback_data = get_feedback_percentages(article["id"])
                article["real_percentage"] = feedback_data["real_percentage"]
                article["fake_percentage"] = feedback_data["fake_percentage"]

            # Fetch related uploads
            related_uploads = [upload for upload in uploads_store if query.lower() in upload["topic"].lower()]
        
        except Exception as e:
            print(f"Error in index route: {str(e)}")
            error_message = str(e)
            
    return render_template("index.html", articles=articles, uploads=related_uploads, error_message=error_message)

@app.route("/feedback", methods=["POST"])
def feedback():
    news_id = request.form.get("news_id")
    is_real = request.form.get("is_real")

    # Update feedback store
    if news_id not in feedback_store:
        feedback_store[news_id] = {"real": 0, "fake": 0}
    feedback_store[news_id][is_real] += 1
    save_feedback()  # Save to JSON file

    return redirect(url_for("index"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        topic = request.form.get("topic")
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Add upload to store
            uploads_store.append({"filename": filename, "topic": topic})
            save_uploads()  # Save to JSON file

            return redirect(url_for("upload"))

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
            