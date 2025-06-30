import nltk
nltk.download('punkt_tab')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



training_data = [
    ("hello", "Hi there! How can I help you?"),
    ("how are you", "I'm doing well, thank you!"),
    ("what is the weather today", "I am sorry, I cannot provide real-time weather information."),
    ("goodbye", "Goodbye! Have a great day."),
    ("tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("what is india's capital", "India's capital is New Delhi.")
]

vectorizer = TfidfVectorizer()
X_train = [text for text, _ in training_data]
y_train = [label for _, label in training_data]
vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    intent = model.predict(user_input_vectorized)[0]
    for text, label in training_data:
        if label == intent:
            return label
    return "I'm sorry, I don't understand. Could you please rephrase your question?"

print("Chatbot: Hi there! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
