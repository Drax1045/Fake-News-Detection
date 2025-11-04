import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Load cleaned dataset
print("📂 Loading data...")
data = pd.read_csv('data/cleaned_data.csv')

# 2️⃣ Text Preprocessing
print("🧹 Cleaning text data...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(preprocess)

# 3️⃣ Feature Extraction
print("🧮 Extracting features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text']).toarray()
y = data['label']

# 4️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Model Training
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6️⃣ Evaluation
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save Model
joblib.dump(model, 'src/fake_news_model.pkl')
joblib.dump(vectorizer, 'src/tfidf_vectorizer.pkl')
print("\n💾 Model and vectorizer saved successfully!")
