import sys
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix

def accuracy(clf, X, y):
	# Given a classifier, test set, and corresponding labels, print the accuracy
	print('The accuracy is: ', clf.score(X, y))

# def print_top_features(clf, top_n, feature_names): #Fei-Tzin Lee advised to use selectkbest on the test set during office hours
# 	print('Top Features')
# 	for index in reversed(np.argsort(clf.coef_[0])[-top_n:]):
# 	       print(feature_names[index])

def print_top_features(X, y, top_n, feature_names):
	'''
	Print the top features for given Data

	args:
		X (sparse matrix) : Feature matrix
		y (list[str]) : labels corresponding to X
		top_n (int) : number of features to print
		feature_names (list(str)) : Indexed feature names

	'''
	feature_indices = SelectKBest(f_classif, k = top_n).fit(X, y).get_support(indices = True) # Use select k best to find top features
	print('Top Features')
	for index in feature_indices:
		print(feature_names[index])

def get_confusion_matrix(clf, X_test, y_test, labels):
	#Given a classifier, test feature matrix, and labels, predict labels and print the confusion matrix
	y_pred = clf.predict(X_test)
	print('Contingency Table\n', confusion_matrix(y_test, y_pred, labels))

def analyze_main(trained_model, X_test, y_test, feature_names):
	'''
	Main functionality for analyze.py

	args:
		trained_model (clf) : Trained model for HW1
		X_test (sparse matrix) : Feature matrix 
		y_test (list[str]) : testing labels 
		labels (list[str]) : class names
	'''
	accuracy(trained_model, X_test, y_test) 
	print()
	print_top_features(X_test, y_test, 20, feature_names)
	print()
	get_confusion_matrix(trained_model, X_test, y_test, ['democrat', 'republican'])
	

if __name__ == '__main__':
	#Extract model and vectorized data from command line print proper information.
	trained_model = pickle.load(open(sys.argv[1], 'rb'))
	X_test, y_test, feature_names = pickle.load(open(sys.argv[2], 'rb'))
	analyze_main(trained_model, X_test, y_test, feature_names)