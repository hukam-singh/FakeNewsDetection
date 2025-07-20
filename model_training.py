import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0  # Fake
real["label"] = 1  # Real

# Keep only text and label
fake = fake[['text', 'label']].dropna()
real = real[['text', 'label']].dropna()

# Balance datasets
min_len = min(len(fake), len(real))
fake = fake.sample(min_len, random_state=42)
real = real.sample(min_len, random_state=42)

# Combine and shuffle
data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Print a sample real news
print("\n🔍 Real News Sample:\n", real.iloc[0]['text'])

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model and vectorizer saved successfully!")
