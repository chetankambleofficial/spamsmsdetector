#Spam-Sms-Detector

This project is a machine learning-based web application designed to classify SMS and email messages as either "Spam" or "Not Spam." It uses Natural Language Processing (NLP) and Machine Learning techniques to preprocess text and predict spam using a trained model. The app is built with Streamlit for an intuitive, interactive interface.

![image](https://github.com/user-attachments/assets/9ec02955-d1fb-4cb9-9f35-391f5741515a)
loading datasets
![image](https://github.com/user-attachments/assets/5c9d5983-eed5-4ef5-b0b7-218d86747da2)
cleaning dataset
![image](https://github.com/user-attachments/assets/94b0ac39-8b2e-4f40-a59e-556f704560fb)




Features
Real-Time Predictions: Classify messages in real-time as spam or not spam using a simple interface.
Text Preprocessing: Converts text to lowercase, removes punctuation and stopwords, and stems words using NLP techniques.
Machine Learning Model: A trained model that predicts whether a message is spam based on its content.
Streamlit Web App: Interactive and user-friendly interface for inputting messages and getting predictions instantly.
Installation and Usage
Prerequisites
Make sure you have the following installed:

Python 3.6+
pip (Python package installer)
Step-by-Step Setup
Clone the repository:


git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
Install dependencies: Use the requirements.txt file to install the required Python libraries:

bash

pip install -r requirements.txt
Run the application: Use the Streamlit CLI to run the app:

bash

streamlit run app.py
Open the app in your browser: The app will run locally at http://localhost:8501/ by default. Open this URL in your browser to interact with the classifier.

Project Structure
bash
Copy code
sms-spam-classifier/
│
├── app.py                   # Main Streamlit app
├── vectorizer.pkl            # Pre-trained TF-IDF vectorizer
├── model.pkl                 # Pre-trained machine learning model
├── requirements.txt          # Required Python libraries
└── README.md                 # Project documentation
How It Works
Input: Users enter a message into the text box provided by the Streamlit interface.
Text Preprocessing:
The text is converted to lowercase.
Tokenization is applied using NLTK.
Stopwords and punctuation are removed.
The remaining text is stemmed using the PorterStemmer.
Vectorization: The preprocessed text is converted into a numerical form using TF-IDF Vectorization.
Prediction: The trained model predicts whether the message is spam or not spam based on the vectorized input.
Output: The result is displayed as either "Spam" or "Not Spam" on the app interface.
Technology Stack
Python
Streamlit for the web app interface
scikit-learn for TF-IDF vectorization and model building
NLTK (Natural Language Toolkit) for text preprocessing
Pickle for model and vectorizer serialization
Model Details
The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer converts the input text into numerical vectors that represent the importance of words in relation to the corpus.
A machine learning model (e.g., Naive Bayes or Logistic Regression) is trained on a dataset of spam and non-spam messages.
The saved model (model.pkl) is used to predict if an input message is spam or not based on its content.
Example
Here’s a sample prediction flow:

Input: "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize!"

Output: Spam

Input: "Hi, are we still meeting for lunch tomorrow?"

Output: Not Spam

Contributing
Contributions are welcome! If you want to improve this project, feel free to submit a pull request or open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Streamlit for providing an easy-to-use interface for deploying machine learning models.
scikit-learn and NLTK for powerful libraries to handle text preprocessing and model development.
Special thanks to the contributors and open-source libraries that made this project possible.
