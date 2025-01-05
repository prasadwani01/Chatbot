# AI-Powered Student Performance Chatbot

A comprehensive AI-based chatbot that assists students in understanding their academic performance, attendance, and overall progress. The chatbot integrates **Natural Language Processing (NLP)**, **Sentiment Analysis**, **Multilingual Translation**, and personalized responses using **machine learning models** and **MongoDB** for database management.

---

## 📌 Features

- **Performance Feedback**: Provides subject-wise marks and overall performance.
- **Attendance Insights**: Displays attendance percentages with empathetic responses based on sentiment analysis.
- **Sentiment Analysis**: Adapts responses based on the user's emotional tone (positive, neutral, or negative).
- **Multilingual Support**: Supports real-time translation for inputs in various languages using the Hugging Face Transformer model.
- **Dynamic Data**: Fetches and processes data from MongoDB for real-time updates.
- **Personalized Responses**: Addresses students by name and tailors answers based on their data.

---

## 📂 Project Structure


---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or above
- MongoDB (running locally or on the cloud)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ai-student-chatbot.git
   cd ai-student-chatbot
##Install Dependencies
-pip install -r requirements.txt
##Set Up MongoDB

-Import the student.students.json into your MongoDB database.
-Ensure the database name is student and the collection is students.

##Train the Chatbot Model Run the training script to generate the chatbot_model.h5 file:
-python Train.py

##Run the Chatbot Start the chatbot system:
-python Response_System.py

##Interact with the Chatbot

-Enter your Student Code.
-Choose a language for communication.
-Ask questions about your performance or attendance.

##Technologies Used
-Backend: Python, Flask
-Machine Learning: TensorFlow, Hugging Face Transformers
-NLP Libraries: NLTK, SpaCy, TextBlob
-Database: MongoDB
-Sentiment Analysis: TextBlob

🌐 Multilingual Support
##Supported languages include:

-Spanish (es)
-French (fr)
-Hindi (hi)
-Marathi (mr)
And more via Hugging Face's MarianMT models.
