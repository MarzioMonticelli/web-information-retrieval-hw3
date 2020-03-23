from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD

import sklearn.metrics as skmetric
from sklearn.datasets import load_files

from nltk.corpus import stopwords
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

import pprint as pp

############################################
stemmer = EnglishStemmer()


def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text


######################################################################


data_folder_training_set = "../data/Positve_negative_sentences/Training"
data_folder_test_set = "../data/Positve_negative_sentences/Test"

training_dataset = load_files(data_folder_training_set)
test_dataset = load_files(data_folder_test_set)

print
print "----------------------"
print(training_dataset.target_names)
print "----------------------"
print

# Load Training-Set
X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(
    training_dataset.data,
    training_dataset.target,
    test_size=0.0)

# Load Test-Set
X_train_DUMMY_to_ignore, X_test, Y_train_DUMMY_to_ignore, Y_test = train_test_split(
    test_dataset.data,
    test_dataset.target,
    train_size=0.0)

target_names = training_dataset.target_names

positive_train_counter = 0
negative_train_counter = 0
for el in Y_train:
    if el == 1:
        negative_train_counter += 1
    else:
        positive_train_counter += 1

positive_test_counter = 0
negative_test_counter = 0
for el in Y_test:
    if el == 1:
        negative_test_counter += 1
    else:
        positive_test_counter += 1

print
print "----------------------"
print "Creating Training Set and Test Set"
print
print "Training Set Size"
print(Y_train.shape)
print "Positive ", negative_train_counter
print "Negative ", positive_train_counter
print
print "Test Set Size"
print(Y_test.shape)
print "Positive ", positive_test_counter
print "Negative ", negative_test_counter
print
print("Classes:")
print(target_names)
print "----------------------"

# Vectorization object
vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None,)

n_grams = [(x, y) for x in xrange(1, 2, 1) for y in xrange(2, 4, 1) if x < y]

classifiers = [
    ['KNN', KNeighborsClassifier(n_jobs=1, )],
    ['SGD', SGDClassifier()],
    ['DECISION TREE', DecisionTreeClassifier()],
]

parameters_list = [
    ['KNN', {
        'dec': TruncatedSVD(),
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__ngram_range': n_grams,
        'vect__analyzer': ['word', 'char'],
        'vect__max_df': np.arange(.8, 1., .1),
        'vect__min_df': np.arange(0., .2, .1),
        'vect__binary': [True, False],
        'vect__lowercase': [True, False],
        'vect__sublinear_tf': [True, False],
        'vect__stop_words': [None, stopwords.words("english")],

        'dec__n_components': xrange(10, 15, 2),
        'nbc__n_neighbors': xrange(3, 6, 1),
        'nbc__weights': ['distance', 'uniform'],
    }],
    ['DECISION TREE', {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__ngram_range': n_grams,
        'vect__analyzer': ['word', 'char'],
        'vect__max_df': np.arange(.8, 1., .1),
        'vect__min_df': np.arange(0., .2, .1),
        'vect__binary': [True, False],
        'vect__lowercase': [True, False],
        'vect__sublinear_tf': [True, False],
        'vect__stop_words': [None, stopwords.words("english")],
    }],
    ['SGD', {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__ngram_range': n_grams,
        'vect__analyzer': ['word', 'char'],
        'vect__max_df': np.arange(.8, 1., .1),
        'vect__min_df': np.arange(0., .2, .1),
        'vect__binary': [True, False],
        'vect__lowercase': [True, False],
        'vect__sublinear_tf': [True, False],
        'vect__stop_words': [None, stopwords.words("english")],
    }],
]

count = 0
for classifier in classifiers:

    # With a Pipeline object we can assemble several steps
    # that can be cross-validated together while setting different parameters.
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('dec', None),
        ('nbc', classifier[1]),
    ])

    # Setting parameters.
    # Dictionary in which:
    #  Keys are parameters of objects in the pipeline.
    #  Values are set of values to try for a particular parameter.
    print
    print "Selected parameter for:"
    parameters = None
    for classf, params in parameters_list:
        if classf == classifier[0]:
            parameters = params
            break

    if parameters is None:
        continue

    print classifier[0]
    # Create a Grid-Search-Cross-Validation object
    # to find in an automated fashion the best combination of parameters.
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring=metrics.make_scorer(skmetric.matthews_corrcoef),
                               error_score=0,
                               cv=10,
                               n_jobs=8,
                               pre_dispatch=10,
                               verbose=1)

    # Start an exhaustive search to find the best combination of parameters
    # according to the selected scoring-function.
    print
    grid_search.fit(X_train, Y_train)
    print

    print
    print("Best Estimator:")
    pp.pprint(grid_search.best_estimator_)
    print
    print("Best Parameters:")
    pp.pprint(grid_search.best_params_)
    print
    print("Used Scorer Function:")
    pp.pprint(grid_search.scorer_)
    print
    print("Number of Folds:")
    pp.pprint(grid_search.n_splits_)
    print

    # Let's train the classifier that achieved the best performance,
    # considering the select scoring-function,
    # on the entire original TRAINING-Set
    Y_predicted = grid_search.predict(X_test)

    # Evaluate the performance of the classifier on the original Test-Set
    output_classification_report = metrics.classification_report(
        Y_test,
        Y_predicted,
        target_names=target_names)
    print
    print "----------------------------------------------------"
    print(output_classification_report)
    print "----------------------------------------------------"
    print

    # Compute the confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
    print
    print("Confusion Matrix: True-Classes X Predicted-Classes")
    print(confusion_matrix)
    print

    # Compute the accuracy classification score normalized (best performance 1)
    accuracy_score_normalized = metrics.accuracy_score(
        Y_test,
        Y_predicted,
        normalize=True)
    print
    print("Accurancy Score Normalized: float")
    print(accuracy_score_normalized)
    print

    matthews_correlation_coefficent = metrics.matthews_corrcoef(
        Y_test,
        Y_predicted)

    print
    print("Matthews correlation coefficient : float")
    print(matthews_correlation_coefficent)
    print

    print '*' * 100
