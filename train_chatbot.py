import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('intents.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(filtered)

data['processed'] = data['question'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['processed'], data['intent'], test_size=0.2, random_state=42)

# Build pipeline (Vectorizer + Classifier)
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to predict intent
def predict_intent(text):
    text_processed = preprocess_text(text)
    return model.predict([text_processed])[0]

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Bot: Goodbye! 👋")
        break
    intent = predict_intent(user_input)
    print(f"Bot Intent: {intent}")

