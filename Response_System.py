from textblob import TextBlob
import random
import json
import pymongo
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import spacy
from transformers import MarianMTModel, MarianTokenizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")  # Load SpaCy for NER

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["student"]
collection = db["students"]

# Load necessary files
with open('intents.json') as file:
    intents_json = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Translation function
def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text  # No translation needed
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Helper function: Clean and lemmatize the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

# Helper function: Create a bag-of-words representation
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Helper function: Predict intent using the trained model
def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Fetch student data from MongoDB
def get_student_data(student_code):
    student = collection.find_one({"Student Code": student_code})
    return student

# Extract entities using SpaCy
def extract_entities(message):
    doc = nlp(message)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Helper function for personalization
def personalize_response(response, student_data):
    name = student_data.get("Student Name", "Student")
    return response.replace("{name}", name)

# Sentiment Analysis Function
def analyze_sentiment(message):
    sentiment = TextBlob(message).sentiment
    return sentiment.polarity

# Main function for response generation with sentiment
def get_response_with_sentiment(intents_list, intents_json, student_code, message, lang):
    translated_message = translate_text(message, lang, "en")
    print(f"Translated Message: {translated_message}")

    sentiment_polarity = analyze_sentiment(translated_message)
    print(f"Sentiment Polarity: {sentiment_polarity}")

    intents_list = predict_class(translated_message)
    student_data = get_student_data(student_code)
    if not student_data:
        return "Student data not found. Please check your Student Code."

    if intents_list:
        intent = intents_list[0]['intent']
        for item in intents_json['intents']:
            if item['tag'] == intent:
                if intent == "performance":
                    # Extract marks from the 'subjects' array
                    subjects = student_data.get("subjects", [])
                    if subjects:
                        marks_list = "\n".join([f"{subject['subject']}: {subject['marks']}" for subject in subjects])
                        return f"Hello, {student_data.get('Student Name', 'Student')}! Here are your marks:\n{marks_list}."
                    else:
                        return f"Hello, {student_data.get('Student Name', 'Student')}! I couldn't find your marks in the system. Please ensure they are updated."

                elif intent == "attendance":
                    attendance = student_data.get("attendance", "unknown")
                    if sentiment_polarity < 0:
                        return f"I see you're concerned about your attendance, {student_data.get('Student Name', 'Student')}. It's currently at {attendance}%. Let's work together to improve it!"
                    else:
                        return f"Hello, {student_data.get('Student Name', 'Student')}! Your attendance: {attendance}%."
                else:
                    response = random.choice(item['responses'])
                    return personalize_response(response, student_data)
    return "I'm sorry, I didn't understand that."


# Main chatbot loop
print("Chatbot is running!")
student_code = input("Enter your Student Code: ")
language = input("Enter your language code (e.g., 'es' for Spanish, 'fr' for French, 'en' for English): ")
while True:
    message = input("You: ")
    response = get_response_with_sentiment([], intents_json, student_code, message, language)
    print(f"Chatbot: {response}")
