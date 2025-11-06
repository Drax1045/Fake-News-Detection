from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
import string

# --- Setup and Preprocessing ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Converts text to lowercase, removes punctuation, and removes stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Data Loading and Balancing ---
print("⏳ Loading and Balancing Data...")
real = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

real['label'] = 1
fake['label'] = 0

# 1. Combine data first (before sampling)
data_combined = pd.concat([real, fake])

# 2. Determine minimum length and balance the dataset
min_len = min(len(real), len(fake))
real_balanced = data_combined[data_combined['label'] == 1].sample(min_len, random_state=42)
fake_balanced = data_combined[data_combined['label'] == 0].sample(min_len, random_state=42)

# Create the final, balanced dataset and shuffle it
data = pd.concat([real_balanced, fake_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Create a combined feature (Title + Text) for better features
# Ensure all columns are string type before concatenation and cleaning
data['full_text'] = data['title'].astype(str)

# 4. Clean the combined text (CRITICAL STEP: Done AFTER balancing)
data['full_text'] = data['full_text'].apply(clean_text)
print("✅ Data cleaned and balanced successfully.")

# --- Model Training Pipeline ---
# Train-Test Split using the new 'full_text' column
X_train, X_test, y_train, y_test = train_test_split(data['full_text'], data['label'], test_size=0.2, random_state=42)

# TF-IDF with bi-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("✨ TF-IDF vectorization complete.")

# Logistic Regression (balanced)
model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
model.fit(X_train_tfidf, y_train)
print("🧠 Model training complete.")

# --- Evaluate ---
y_pred = model.predict(X_test_tfidf)
print("\n--- 📊 Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save Model ---
joblib.dump(model, 'src/fake_news_model.pkl')
joblib.dump(vectorizer, 'src/tfidf_vectorizer.pkl')

print("\n💾 Model retrained and saved successfully! You should see much better classification now.")