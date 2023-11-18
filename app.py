# Importing libraries
import difflib
import pickle
from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, app, redirect, render_template, request
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import difflib
import numpy as np
import math
from fuzzywuzzy import fuzz
import requests

app = Flask(__name__)
app.static_folder = 'static'

#<-------- Training and Testing dataset ----------->

# Load the dataset
dataset_path = 'model_dataset.csv'
df = pd.read_csv(dataset_path)

# Ensure that the columns exist in your DataFrame
required_columns = ['question', 'model_answer', 'keywords', 'out_of']
if not all(col in df.columns for col in required_columns):
    raise KeyError(f"Columns {', '.join(required_columns)} must exist in your DataFrame.")

# Handle missing values in the 'model_answer' and 'keywords' columns
df['model_answer'].fillna('', inplace=True)
df['keywords'].fillna('', inplace=True)

# Split the dataset into training and testing sets
train_data, test_data, train_out_of, test_out_of = train_test_split(
    df[['model_answer', 'keywords']], df['out_of'], test_size=0.2, random_state=42
)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['model_answer'] + ' ' + train_data['keywords'])
X_test = vectorizer.transform(test_data['model_answer'] + ' ' + test_data['keywords'])

# Handle missing values in the target variable 'out_of'
train_out_of = train_out_of.replace(np.nan, 0)  # Replace NaN with 0, adjust as needed

# Train a linear SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, train_out_of)


# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(test_out_of, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display additional metrics
# print("Classification Report:")
# print(classification_report(test_out_of, predictions))
# print("Out of Marks: ")
# print(df)

# Image Processing
rcParams['figure.figsize'] = 8, 16
reader = easyocr.Reader(['en', 'hi'])


# Set up a folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pickle file for evaluation
pickle_path = 'nav_test.pickle'
with open(pickle_path, 'rb') as pickle_file:
    clf = pickle.load(pickle_file)

# Define the predict function here
def predict(key, g, q):
    try:
        # Convert key, g, and q to numeric types if they are not already
        key = float(key)
        g = float(g)
        q = float(q)
    except ValueError as e:
        print(f"Error converting to float: {e}")
        return None  # Handle the error gracefully or provide a default value

    # Assuming clf is a trained classifier
    # You need to have a trained classifier (clf) before calling this function
    predicted = clf.predict([[key, g, q]])
    
    # Assuming you want the probability estimates
    accuracy = clf.predict_proba([[key, g, q]])
    
    print("Class:[0-9]: " + str(predicted))
    # print("Out of: " + str(out_of_marks))
    print("Accuracy: " + str(np.max(accuracy)))
    
    return predicted

# Your existing code goes here (including imports and image processing)

# Assuming 'text_str' contains the extracted text
# Convert the extracted text to a DataFrame for easy comparison
result_df = pd.DataFrame({'extracted_text': ['']})

# Add a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    text_str = ''
    most_similar_question, most_similar_model_answer = '', ''
    similarity = 0.0
    out_of_marks, keywords = 0, ''
    predicted_out_of_marks = 0  # Initialize the variable
    result = 0  # Initialize the variable
    grammar, qst = 0, 0  # Initialize grammar and QST variables

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file
            file_name = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.jpg')
            file.save(file_name)

            # Perform image processing
            output = reader.readtext(file_name)
            print(output)

            # Extract and convert only the text elements to a string
            text_list = [item[1] for item in output]
            text_str = ' '.join(text_list)

            # Convert the extracted text to a DataFrame for easy comparison
            result_df = pd.DataFrame({'extracted_text': [text_str]})

            # Merge the extracted text DataFrame with the original dataset on the 'model_answer' and 'keywords' columns
            merged_df = pd.merge(result_df, df[['question', 'model_answer', 'keywords', 'out_of']], how='left',
                                 left_on=['extracted_text'], right_on=['model_answer'])

            # Handle missing values in the target variable 'out_of' in the merged DataFrame
            merged_df['out_of'].fillna(0, inplace=True)  # Replace NaN with 0, adjust as needed

            # Find the most similar question and its corresponding model answer
            most_similar_question, most_similar_model_answer = '', ''
            highest_similarity = 0
            matching_row = None

            for index, row in df.iterrows():
                similarity = difflib.SequenceMatcher(None, str(text_str), str(row['question'])).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_question = row['question']
                    most_similar_model_answer = row['model_answer']
                    matching_row = row  # Store the matching row

            print(f"The most similar question is: {most_similar_question}")
            print(f"The corresponding model answer is: {most_similar_model_answer}")

            if matching_row is not None:
                # Extract 'out_of' and 'keywords' from the matching row
                out_of_marks = matching_row['out_of']
                keywords = matching_row['keywords']
                print(f"Out of Marks: {out_of_marks}")
                print(f"Keywords: {keywords}")
            else:
                print("No matching row found in the dataset.")

            # Calculate similarity between the extracted text and the most similar model answer
            similarity = difflib.SequenceMatcher(None, text_str, most_similar_model_answer).ratio()

            # Set a threshold for considering them as a match
            accuracy_threshold = 0.8  # Adjust the threshold based on your requirements

            # Check if the similarity meets the threshold
            if similarity >= accuracy_threshold:
                print(f"The extracted text and the corresponding model answer are considered a match with {similarity:.2%} similarity.")

                # Print Keywords, Grammar, and QST for the extracted text
                print("Keywords in Extracted Text: ", keywords)

                # GRAMMAR =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                req = requests.get("https://api.textgears.com/check.php?text=" + text_str + "&key=JmcxHCCPZ7jfXLF6")
                no_of_errors = len(req.json()['errors'])

                if no_of_errors > 5 or len(keywords.split()) <= 5:
                    grammar = 0
                else:
                    grammar = 1
                print("Grammar in Extracted Text: ", grammar)

                # QST =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                qst = math.ceil(fuzz.token_set_ratio(most_similar_model_answer, text_str) * 6 / 100)
                print("QST in Extracted Text: ", qst)

                # Use the provided logic for predicting the result
                predicted = predict(len(keywords.split()), grammar, qst)
                result = predicted * out_of_marks / 10
                print(f"Predicted Marks: {float(result):.2f}")
            else:
                print(f"The extracted text and the corresponding model answer are not considered a match.")

    return render_template('index.html', text_str=text_str, most_similar_question=most_similar_question,
                           most_similar_model_answer=most_similar_model_answer, similarity=similarity,
                           out_of_marks=out_of_marks, keywords=keywords, predicted_out_of_marks=predicted_out_of_marks,
                           result=result, grammar=grammar, qst=qst)

# Your existing code goes here

if __name__ == '__main__':
    app.run(debug=True)