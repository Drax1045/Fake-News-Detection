import feedparser
import pandas as pd
import random
import os
import numpy as np  

print("📥 Fetching real news headlines...")


urls = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.feedburner.com/ndtvnews-latest",
    "https://www.thehindu.com/feeder/default.rss"
]

real_news = []
for url in urls:
    feed = feedparser.parse(url)
    for entry in feed.entries[:50]:  # Take up to 50 per source
        real_news.append(entry.title.strip())

# Save REAL news
real_path = os.path.join("data", "True.csv")
pd.DataFrame(real_news, columns=["title"]).to_csv(real_path, index=False)
print(f"✅ Saved {len(real_news)} real news headlines at {real_path}")

print("🧠 Generating fake (synthetic) headlines...")
fake_templates = [
    "Scientists discover cure for aging",
    "Government to distribute free cars next month",
    "Aliens land in Mumbai to negotiate peace deal",
    "Man becomes billionaire overnight using secret app",
    "Chocolate found to boost intelligence by 300%",
    "AI replaces doctors in hospitals worldwide",
    "India launches first hotel on the Moon",
    "NASA confirms aliens using WhatsApp to communicate",
    "School bans homework after AI rebellion",
    "Mumbai underwater after record chocolate rain"
]

fake_news = [random.choice(fake_templates) + f" #{i+1}" for i in range(100)]

# Save FAKE news
fake_path = os.path.join("data", "Fake.csv")
pd.DataFrame(fake_news, columns=["title"]).to_csv(fake_path, index=False)
print(f"✅ Saved {len(fake_news)} fake news headlines at {fake_path}")

print("🎉 Data generation completed! You can now retrain the model.")
