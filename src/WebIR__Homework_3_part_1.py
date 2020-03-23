from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skmetric
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

import pprint as pp
import numpy as np
from nltk.corpus import words

"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('words')
# """

############################################
stemmer = EnglishStemmer()


def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text


######################################################################

data_folder_training_set = "../data/Ham_Spam_comments/Training"
data_folder_test_set = "../data/Ham_Spam_comments/Test"

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

ham_train_counter = 0
spam_train_counter = 0
for el in Y_train:
    if el == 1:
        spam_train_counter += 1
    else:
        ham_train_counter += 1

ham_test_counter = 0
spam_test_counter = 0
for el in Y_test:
    if el == 1:
        spam_test_counter += 1
    else:
        ham_test_counter += 1

print
print "----------------------"
print "Creating Training Set and Test Set"
print
print "Training Set Size"
print(Y_train.shape)
print "Spam ", spam_train_counter
print "Ham ", ham_train_counter
print
print "Test Set Size"
print(Y_test.shape)
print "Spam ", spam_test_counter
print "Ham ", ham_test_counter
print
print("Classes:")
print(target_names)
print "----------------------"

# dictionary
# en_dict = set(words.words())
# print "Vocabulary length: ", str(len(en_dict))

n_grams = [(x, y) for x in xrange(1, 3, 1) for y in xrange(1, 10, 1) if x < y]

# Vectorization object
# lowercase = True, because stopwords are lowercase
vectorizer = TfidfVectorizer(strip_accents=None,
                             preprocessor=None,)

decompositor = TruncatedSVD()

nbc = KNeighborsClassifier(n_jobs=1,)

"""    
    n_neighbors=3,
    weights='distance',
"""

# With a Pipeline object we can assemble several steps
# that can be cross-validated together while setting different parameters.
pipeline = Pipeline([
    ('vect', vectorizer),
    ('dec', decompositor),
    ('nbc', nbc),
])

# Setting parameters.
# Dictionary in which:
#  Keys are parameters of objects in the pipeline.
#  Values are set of values to try for a particular parameter.
parameters = {
    'vect__tokenizer': [None, stemming_tokenizer],
    'vect__analyzer': ['word', 'char'],
    'vect__ngram_range': n_grams,
    'vect__max_df': np.arange(.6, 1, .1),
    'vect__min_df': np.arange(0., .3, .1),
    'vect__binary': [True, False],
    'vect__lowercase': [True, False],
    'vect__sublinear_tf': [True, False],
    'vect__stop_words': [None, stopwords.words("english")],
    'dec__n_components': xrange(1, 11, 1),

    'nbc__n_neighbors': xrange(1, 11, 1),
    'nbc__weights': ['uniform', 'distance'],
}

# Create a Grid-Search-Cross-Validation object
# to find in an automated fashion the best combination of parameters.
grid_search = GridSearchCV(pipeline, parameters,
                           pre_dispatch=10,
                           refit=True,
                           scoring=metrics.make_scorer(skmetric.matthews_corrcoef),
                           error_score=0,
                           cv=10, n_jobs=8, verbose=1, )

# Start an exhaustive search to find the best combination of parameters
# according to the selected scoring-function.
print
grid_search.fit(X_train, Y_train)
print

# Print results for each combination of parameters.print    ['DECISION TREE', DecisionTreeClassifier(presort=True)],
number_of_candidates = len(grid_search.cv_results_['params'])
""" print("Results:")
for i in range(number_of_candidates):
    print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
          (grid_search.cv_results_['params'][i],
           grid_search.cv_results_['mean_test_score'][i],
           grid_search.cv_results_['std_test_score'][i]))
"""
print
print("Best Estimator:")
pp.pprint(grid_search.best_estimator_)
print
print("Best Parameters:")
pp.pprint(grid_search.best_params_)
print
print("Used Scorer Function:")
pp.pprint(grid_search.scorer_)
print("Number of Folds:")
pp.pprint(grid_search.n_splits_)
print
print ("Best Score")
pp.pprint(grid_search.best_score_)
print

# Let's train the classifier that achieved the best performance,
# considering the select scoring-function,
#  on the entire original TRAINING-Set
Y_predicted = grid_search.predict(X_test)

# Evaluate the performance of the classifier on the original Test-Set
output_classification_report = metrics.classification_report(
    Y_test,
    Y_predicted,
    target_names=target_names,
    digits=2, )
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
accuracy_score_normalized = metrics.accuracy_score(Y_test, Y_predicted, normalize=True)
print
print("Accurancy Score Normalized: float")
print(accuracy_score_normalized)
print

matthews_correlation_coefficent = metrics.matthews_corrcoef(Y_test, Y_predicted)

print
print("Matthews correlation coefficient : float")
print(matthews_correlation_coefficent)
print

print '*' * 100
