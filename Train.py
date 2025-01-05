import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from nlpaug.augmenter.word import SynonymAug
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model for SynonymAug
spacy_model = spacy.load("en_core_web_sm")

# Custom tokenizer for spaCy integration
def custom_tokenizer(sentence):
    return [token.text for token in spacy_model(sentence)]

# Initialize lemmatizer and SynonymAug
lemmatizer = WordNetLemmatizer()
aug = SynonymAug(aug_src='wordnet', tokenizer=custom_tokenizer)

# Load intents.json file
with open('intents.json') as file:
    intents = json.load(file)

# Augment patterns
for intent in intents["intents"]:
    augmented_patterns = []
    for pattern in intent["patterns"]:
        augmented_patterns.append(pattern)  # Add original pattern
        try:
            augmented = aug.augment(pattern, n=3)  # Generate 3 augmented patterns
            augmented_patterns.extend(augmented)
        except Exception as e:
            print(f"Augmentation error for pattern '{pattern}': {e}")
    intent["patterns"] = list(set(augmented_patterns))  # Remove duplicates

# Prepare patterns and tags
patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Create vocabulary (words) and unique tags (classes)
words = []
classes = sorted(set(tags))
ignore_words = ['?', '!', '.', ',']

for pattern in patterns:
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)

words = sorted(set([lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]))

# Define the bag_of_words function
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return bag

# Generate training data
train_x = [bag_of_words(pattern, words) for pattern in patterns]
train_y = [classes.index(tag) for tag in tags]
train_y = tf.keras.utils.to_categorical(train_y, num_classes=len(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Convert to NumPy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print("Model training complete and saved as chatbot_model.h5!")
