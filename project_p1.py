# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2023
# Project Part 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
#
# This code will be graded automatically using Gradescope.
# =========================================================================================================



import pandas as pd
import numpy as np
import pickle as pkl
import nltk
import time
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score




# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)


# Function: preprocessing.  See project statement for more details.
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    # Write your code here:

    return modified_input


# Function: vectorize_train.  See project statement for more details.
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    # Update the vectorizer and tfidf_train
    vectorizer  = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    # Check whether the given token is in word2vec
    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = []
    avg_embed = np.zeros(300, )     # Re-initialize the embedding array

    # Get token from the user input:
    tokens = get_tokens(user_input)
    for token in tokens:
        w2v_convs = w2v(word2vec, token)            # Convert token to the w2v model style
        # Checking whether the w2v convert to the numeric-list or not
        if (not isinstance(w2v_convs, list)):
            w2v_convs = w2v_convs.tolist()
        embedding.append(w2v_convs)                 # Add the appropriate to the embedding array

    # Update the original embedding with its average using the np.mean
    if len(embedding) > 0:
        avg_embed = np.mean(embedding, axis=0)

    # print("LOCAL TESSTING, ", avg_embed)
    return avg_embed


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = GaussianNB()
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using TFIDF
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    # Convert the training data to a numpy_array and apply the labels
    return model.fit(tfidf_train.toarray(), training_labels)


# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    # Convert training_document to embedding w2v
    embedding_lst = []
    for document in training_documents:
        word2vec_converter = string2vec(word2vec, document)
        embedding_lst.append(word2vec_converter)

    # Casting embedding to a matrix array and return fit model
    return model.fit(np.array(embedding_lst), training_labels)


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Vectorizing to normalize each word from the test document
    # Convert to array type to apply for model prediction
    model_test_vector = vectorizer.transform(test_documents).toarray()

    # Predict labels
    label_predicts = model.predict(model_test_vector)

    # Statistical performing calculation
    # Update variables above to appropriate type
    precision = precision_score(test_labels, label_predicts)
    recall    = recall_score(test_labels, label_predicts)
    f1        = f1_score(test_labels, label_predicts)
    accuracy  = accuracy_score(test_labels, label_predicts)

    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Convert training_document to embedding w2v
    model_test_vector = []
    for document in test_documents:
        word2vec_converter = string2vec(word2vec, document)
        model_test_vector.append(word2vec_converter)

    # Predict labels
    label_predicts = model.predict(model_test_vector)

    # Statistical performing calculation
    # Update variables above to appropriate type
    precision = precision_score(test_labels, label_predicts)
    recall = recall_score(test_labels, label_predicts)
    f1 = f1_score(test_labels, label_predicts)
    accuracy = accuracy_score(test_labels, label_predicts)

    return precision, recall, f1, accuracy


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# project questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    print("*************** Loading data & processing *****************")
    # Load the dataset
    print("Loading dataset.csv....")
    documents, labels = load_as_list("dataset.csv")
    

    # Load the Word2Vec representations so that you can make use of it later
    print("Loading Word2Vec representations....")
    word2vec = load_w2v(EMBEDDING_FILE)

    # Compute TFIDF representations so that you can make use of them later
    print("Computing TFIDF representations....")
    vectorizer, tfidf_train = vectorize_train(documents)

    # print(tfidf_train)
    # exit(0)


    print("\n**************** Training models ***********************")
    # Instantiate and train the machine learning models
    print("Instantiating models....")
    nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models()
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()

    print("Training Naive Bayes models....")
    start = time.time() # This will help you monitor training times (useful once training functions are implemented!)
    nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    end = time.time()
    print("Naive Bayes + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    end = time.time()
    print("Naive Bayes + w2v trained in {0} seconds".format(end - start))

    print("Training Logistic Regression models....")
    start = time.time()
    logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    end = time.time()
    print("Logistic Regression + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    end = time.time()
    print("Logistic Regression + w2v trained in {0} seconds".format(end - start))

    print("Training SVM models....")
    start = time.time()
    svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    end = time.time()
    print("SVM + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    end = time.time()
    print("SVM + w2v trained in {0} seconds".format(end - start))

    print("Training Multilayer Perceptron models....")
    start = time.time()
    mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    end = time.time()
    print("Multilayer Perceptron + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)
    end = time.time()
    print("Multilayer Perceptron + w2v trained in {0} seconds".format(end - start))

    # Uncomment the line below to test out the w2v() function.  Make sure to try a few words that are unlikely to
    # exist in its dictionary (e.g., "covid") to see how it handles those.
    #print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    # Test the machine learning models to see how they perform on the small test set provided.
    # Write a classification report to a CSV file with this information.
    print("\n***************** Testing models ***************************")
    test_documents, test_labels = load_as_list("test.csv")  # Loading the dataset

    models_tfidf = [nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf]
    models_w2v = [nb_w2v, logistic_w2v, svm_w2v, mlp_w2v]
    model_names = ["Naive Bayes", "Logistic Regression", "SVM", "Multilayer Perceptron"]

    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row

    i = 0
    while i < len(models_tfidf): # Loop through models
        print("Making predictions for " + model_names[i] + "....")
        p, r, f, a = test_model_tfidf(models_tfidf[i], vectorizer, test_documents, test_labels)
        if models_tfidf[i] is None:  # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i] + " + TFIDF", "N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i] + " + TFIDF", p, r, f, a])

        p, r, f, a = test_model_w2v(models_w2v[i], word2vec, test_documents, test_labels)
        if models_w2v[i] is None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i]+" + w2v","N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i]+" + w2v", p, r, f, a])
        i += 1
    outfile.close()


    print("\n************** Beginning chatbot execution *******************")

    # Display a welcome message to the user, and accept a user response of arbitrary length
    user_input = input("Welcome to the CS 421 chatbot!  What do you want to talk about today?\n")

    # Predict user's sentiment
    w2v_test = string2vec(word2vec, user_input) # This assumes you're using Word2Vec representations

    label = None

    if mlp_w2v is not None:
        label = mlp_w2v.predict(w2v_test.reshape(1, -1)) # This assumes you're using mlp_w2v; feel free to update

        if label == 0:
            print("Hmm, it seems like you're feeling a bit down.")
        elif label == 1:
            print("It sounds like you're in a positive mood!")
        else:
            print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
