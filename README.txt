Credit Card Fraud Detection System
Developer: Praneeth Varma Danthuluri

📌 Project Overview This project focuses on creating a robust system to increase the security of financial transactions by identifying fraudulent activity using machine learning. It analyzes historical transaction data to differentiate between honest and dishonest transactions.

🛠️ Machine Learning Models The project implements and compares three primary algorithms:

Logistic Regression: Used for its simplicity and transparency in providing probabilistic baseline alerts.

Random Forest Classifier: An ensemble method that handles high-dimensional data and achieves high accuracy.

Decision Tree Classifier: Employed for its interpretability and ability to trace logical rules for classification.


Shutterstock
📊 Methodology

Handling Class Imbalance: Utilized techniques like SMOTE (Synthetic Minority Over-sampling Technique) to ensure the models effectively capture rare fraudulent instances.

Optimization: Employed Cross-Validation and Grid Search for hyperparameter tuning.

Performance Indicators: Models were evaluated based on accuracy, precision, recall, and F1-score.

📂 Dataset The project utilizes a simulated Credit Card Transactions dataset covering the period from January 2019 to December 2020.

Link: https://www.kaggle.com/datasets/kartik2112/fraud-detection

🚀 Setup & Installation (How to Run) Follow these steps to load and run the project:

Google Drive Setup (Jupyter Notebook)

Create a folder named "credit_card_fraud_detection" in your Google Drive.

Upload the "Final_code.ipynb" file into that folder.

Create a sub-folder named "Data" inside that folder and upload your dataset CSV file.

Flask GUI (Web Interface)

Unzip the project folder on your local system.

Ensure "credit_card_fraud_detection" is the main folder.

Open the "Flask" folder in Visual Studio Code.

Install all required libraries via the terminal using the command: pip install -r requirements.txt

Run the application using the command: python app.py