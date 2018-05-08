from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from classify import *
from analyze import *

def initial_model(ngrams, train_corpus, test_corpus, y_train, y_test, clf):
	'''
	Create the vectorizer for each model, train the classifier, and output results

	args:
		ngrams (tuple) : Number of n grams to use
		train_corpus (list[str]) : List of training data tweets 
		test_corpus (list[str]) : List of testing data tweets 
		y_train (list(str)) : Training set labels
		y_test (list(str)) : Testing set labels
		clf : classifier to be trained

	'''
	vectorizer = CountVectorizer(ngram_range=ngrams)
	X_train = vectorizer.fit_transform(train_corpus)
	X_test  = vectorizer.transform(test_corpus)
	clf.fit(X_train, y_train)
	analyze_main(clf, X_test, y_test, vectorizer.get_feature_names())

if __name__ == '__main__':
	train_corpus, y_train = extract_corpus('train_newline.txt')
	test_corpus, y_test = extract_corpus('dev_newline.txt')

	model_type = ['Unigrams', 'Bigrams', 'Trigrams']
	best_classifier_by_model = [MultinomialNB(alpha = 0.1, fit_prior = False), MultinomialNB(alpha = 0.3, fit_prior = False), MultinomialNB(alpha = 0.9, fit_prior = True)]

	for i in range(3): # Iterate through various n-grams and train the best model for each, found by the grid search below
		print('******************** Model {0} : {1} **************************'.format(i+1, model_type[i]))
		initial_model((i+1,i+1), train_corpus, test_corpus, y_train, y_test, best_classifier_by_model[i])
		print()

	print('************** Model 4 : Best Model (Unigrams, MultinomialNB) ******************')
	# Train our best model
	vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words = "english")
	X_train = add_features(vectorizer.fit_transform(train_corpus), get_features(train_corpus))
	X_test  = add_features(vectorizer.transform(test_corpus), get_features(test_corpus))
	clf = train_best_model(X_train, y_train)
	save_best_model(clf)
	#Create list of feature names
	feature_names = vectorizer.get_feature_names()
	feature_names.extend(['contains_emojis', 'num_emojis', 'tweet_length', 'num_@', 'num_cap_words', 'num_caps'])
	analyze_main(clf, X_test, y_test, feature_names)
	#Save vectorized test set
	pickle.dump((X_test, y_test, feature_names), open('test', 'wb'))

'''
GRID SEARCH TO FIND OPTIMAL PARAMETERS FOR EACH MODEL

def get_best_model(X_train, y_train, X_test, y_test, estimators, parameters):
    best_accuracy = 0
    for i in range(len(parameters)):
        clf = GridSearchCV(estimator=estimators[i], param_grid=parameters[i], cv = 10, n_jobs = -1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(accuracy, clf.best_estimator_)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = 
            clf
    return best_model
	unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
	X_train_unigram = add_features(unigram_vectorizer.fit_transform(train_corpus), get_features(train_corpus))
	X_test_unigram = add_features(unigram_vectorizer.transform(test_corpus), get_features(test_corpus))

	bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
	X_train_bigram = bigram_vectorizer.fit_transform(train_corpus)
	X_test_bigram = bigram_vectorizer.transform(test_corpus)

	trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))
	X_train_trigram = trigram_vectorizer.fit_transform(train_corpus)
	X_test_trigram = trigram_vectorizer.transform(test_corpus)

	estimators = [MultinomialNB(), LogisticRegression(), svm.SVC()]

	mnb_parameters = [
	    {'alpha': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], fit_prior : [True, False]}
	]
	lrg_parameters = [
	    {'penalty' : ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'fit_intercept': [True,  False]},
	]
	swm_parameters = [
	    {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
	    {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}
	]

	parameters = [mnb_parameters, lrg_parameters, swm_parameters]
	ngrams = [[X_train_unigram, X_test_unigram], [X_train_bigram, X_test_bigram], [X_train_trigram, X_test_trigram]]

	best_models = [get_best_model(ngram[0], y_train, ngram[1], y_test, estimators, parameters) for ngram in ngrams]
'''