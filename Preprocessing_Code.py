import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Preprocess data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words
# Use your current preprocessing to generate `words`
words = sorted(set([lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]))
words = sorted(set(words))
classes = sorted(set(classes))

# Save preprocessed data
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

print("words.pkl and classes.pkl have been created!")
