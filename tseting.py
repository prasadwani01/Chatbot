import nltk
from nltk.tag import pos_tag

# Set the explicit path
nltk.data.path = ['C:/Users/prasa/AppData/Roaming/nltk_data']

# Test POS tagging
print(pos_tag(["This", "is", "a", "test"]))
