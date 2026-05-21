# phishing-detector-cameroon
AI-Based Phishing Detection System for Financial Transaction Fraud Prevention in Cameroon

Case Study: MTN Mobile Money and Orange Money

Overview

Financial technology has transformed digital payments and money transfer services in Cameroon. Platforms such as MTN Group Mobile Money and Orange S.A. Orange Money have significantly improved access to financial services, especially for individuals without traditional banking access. However, the rapid growth of mobile financial transactions has also increased the prevalence of cyber fraud and phishing attacks targeting users.

This project presents the Design and Implementation of an AI-Based Phishing Detection System for Financial Transaction Fraud Prevention in Cameroon, focusing on the local context and challenges affecting users of MTN Mobile Money and Orange Money. The system applies Artificial Intelligence and Machine Learning techniques to detect suspicious patterns, fraudulent SMS messages, phishing links, and transaction behaviors that may indicate fraud attempts.

The study is motivated by the increasing reports of financial scams in Cameroon, including fake promotional messages, account verification scams, social engineering attacks, and unauthorized transaction attempts.

Problem Statement

Cameroon has experienced a rise in mobile money adoption alongside increasing cyber-enabled financial fraud. Fraudsters exploit weaknesses in user awareness and communication channels through phishing messages and deceptive transaction requests.

Common attacks include:
	•	Fake MTN or Orange promotional messages
	•	Fraudulent account verification requests
	•	Fake customer service communications
	•	SMS and URL phishing attacks
	•	Social engineering scams
	•	Unauthorized transaction prompts

Current prevention methods rely heavily on manual reporting and user vigilance, creating a need for intelligent automated fraud detection systems capable of identifying threats in real time.

Research Aim

To design and implement an AI-driven phishing detection system capable of identifying potentially fraudulent mobile financial activities affecting users in Cameroon.

Objectives

The project seeks to:
	•	Study phishing and financial fraud trends in Cameroon
	•	Analyze fraud patterns affecting MTN Mobile Money and Orange Money users
	•	Collect and preprocess datasets containing phishing indicators
	•	Design machine learning models for fraud prediction
	•	Build a detection system capable of classifying suspicious activities
	•	Evaluate model performance using standard metrics
	•	Propose recommendations for improving financial cybersecurity awareness

Research Questions
	1.	What phishing techniques are commonly used against mobile money users in Cameroon?
	2.	How can Artificial Intelligence improve fraud detection within mobile financial platforms?
	3.	Which machine learning algorithms provide the highest accuracy for phishing detection?
	4.	Can a localized AI solution improve transaction security for Cameroonian users?

Technologies and Tools

Programming
	•	Python
	•	JavaScript
	•	HTML/CSS

Machine Learning Libraries
	•	Scikit-learn
	•	TensorFlow
	•	Pandas
	•	NumPy
	•	NLTK

Backend
	•	Flask / FastAPI

Database
	•	MySQL
	•	MongoDB

Visualization
	•	Matplotlib
	•	Power BI

Development Environment
	•	Jupyter Notebook
	•	VS Code

Dataset Sources

Possible datasets may include:
	•	Public phishing datasets
	•	SMS spam datasets
	•	Transaction behavior datasets
	•	Locally collected anonymous survey data from Cameroon
	•	Synthetic fraud transaction datasets

Potential features:

Feature	Description
Message Content	SMS text
Sender Information	Phone number patterns
URL Presence	Embedded suspicious links
Keywords	Scam-related words
Transaction Frequency	Number of activities
Transaction Amount	Financial behavior patterns
Device Behavior	Login or usage anomalies

Machine Learning Models

The following algorithms may be implemented and compared:
	•	Logistic Regression
	•	Naive Bayes
	•	Decision Trees
	•	Random Forest
	•	Support Vector Machine (SVM)
	•	Long Short-Term Memory (LSTM)
	•	Deep Neural Networks

Performance Metrics

Evaluation metrics include:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix
	•	ROC-AUC
  
Expected Outcomes

The system should:

- Detect phishing attempts automatically
- Reduce fraudulent financial activities
- Improve user security awareness
- Provide real-time risk assessment
- Support financial fraud prevention efforts in Cameroon

Project Structure

AI-Phishing-Detection-Cameroon/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│
├── models/
│
├── app/
│   ├── dashboard.py
│   ├── api.py
│
├── results/
│
├── docs/
│
├── requirements.txt
│
└── README.md



Installation

Clone the repository:

git clone https://github.com/yourusername/AI-Phishing-Detection-Cameroon.git

Move into the project folder:

cd AI-Phishing-Detection-Cameroon

Install dependencies:

pip install -r requirements.txt

Run application:

python app.py


Future Improvements
	•	Integration with live SMS streams
	•	Real-time transaction monitoring
	•	Deployment as a mobile application
	•	AI-powered chatbot for fraud reporting
	•	Integration with telecom APIs
	•	Multilingual support (English, French, and local languages)


Local Relevance to Cameroon

This research specifically addresses cybersecurity challenges affecting digital finance users in Cameroon. The increasing use of mobile money systems and the growth of online fraud require solutions adapted to local realities, user behavior, and fraud techniques.

By focusing on MTN Mobile Money and Orange Money, the project aims to contribute practical solutions toward safer financial transactions and stronger digital trust in Cameroon.


Author

Name: TCHATCHOUA NGASSAM TRESOR LARRY
Department of ICT / CYBERSECURITY
Institution: THE ICT UNIVERSITY
Country: Cameroon

“Artificial Intelligence can become a powerful tool for securing financial transactions and reducing fraud in Cameroon.”
