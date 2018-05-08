# Political_Affiliation_Predictor
Bag of words based machine learning model with custom engineered features to predict a user’s political affiliation given a single tweet

* The classify.py file that takes the train and test files as command line arguments (i.e., can be run as ‘python classify.py train.txt test.txt’), trains and saves the best-scoring model to a file, and tests that model on the test set and prints its accuracy
* The analyze.py file that takes a trained model and the vectorized test data as command line arguments (e.g., can be run as ‘python analyze.py model.pkl test’), and prints the top 20 features and a contingency table to the console.
 * A wrapper, main.py, that runs both files to print the following to the console: the accuracy of your best-trained model, and the accuracy, top 20 features, and contingency table for each required model.

## Data

Set of tweets labeled as democrat or republican. These tweets were posted during the 2014 midterm election and were self-labeled using a website called WeFollow where people post under either democrat or republican (this website is no longer live). The data files are formatted with one tweet per line, with one line containing the tweet text, a tab, and the label text (i.e. “democrat” or “republican”).

## Technology Used
- Python 3

## Features
Special Features:
* contains_emojis (boolean) : True if the tweet contained at least one emoji, false otherwise
* num_emojis (int) : the number of emojis in the tweet
* tweet_length (int) : Number of characters in the tweet, which can capture word lengthening
* num_at (int) : Number of @ characters used in the tweet, indicating the tweet is directed at one or more other users or referencing them.
* num_cap_words (int) : Number of capital words in a tweet
* num_caps (int) : Number of capital letters in a tweet

The rest of the features are unigram frequencies generated by CountVectorizer, removing stop words.

Limitations of this classifier include inability to detect sarcasm and sentiment, which in some cases leads to incorrect classification.

## Justification

* My best model used unigram features with a Multinomial Naïve Bayes classifier (discovered by an initial grid search for all 9 model combinations). The Multinomial Naïve Bayes classifier was the chosen Naïve Bayes classifier because it can model the information of word frequencies, thus being a good choice for working with text data. Unigrams seem to work the best because the probability of seeing bigrams and trigrams becomes very low, and we do not actually have a very large amount of training data. If we had even more data, we might start to see improvement by adding bigram or trigram features.

* Through grid searching, I found the optimal alpha hyper parameter, which determines the additive smoothing for the model to be 0.2. By using some additive smoothing, our model will assign non-zero probabilities to words that do not occur, which would have 0 probability otherwise and may then not generalize as well to a tweet which contains words not previously seen. Similarly, the fit_prior hyperparameter, which determines whether to learn class prior probabilities or not was optimal at False, indicating that a uniform prior is used.

- When creating the unigram features, stop words were removed. Stop words contain common words that would receive very high probabilities in the Naïve Bayes Classifier and can skew the model. Their removal increased accuracy as expected.

* Other than the unigram features from the CountVectorizer, I have included 6 additional features in which each feature at least slightly improves the model’s performance (seen during both cross validation and on the dev set). The first feature was a boolean value representing if the tweet contained at least one emoji. I considered the Unicode characters of the top few hundred emojis. I found that in the training set, 557 of the twitter users used at least one of those emojis. 399 of them were republicans and 158 were democrats. Since the ratio of republicans to democrats is split pretty evenly in the training set, a tweet containing at least one emoji can help indicate the user’s political affiliation in many cases where less text is present but emojis are (higher chance of republican if an emoji is used). The next feature I included was the number of emojis used. Just because Republican’s tweets seem more likely to contain an emoji than a democrat’s tweets, one group may tend to use multiple emojis more than the other. I found that Republicans use an average of 1.78 emojis, while democrats use an average of 1.85 emojis, which indicates if there are multiple emojis in a tweet, it may help lead to a democratic affiliation. As mentioned, with each feature the accuracy improved during cross validation and the dev set. The next feature I used was tweet length, which is the number of character in the tweet. This can help capture word lengthening as well. Looking at the distribution of tweet length, I found that on average democrats use slightly more characters than Republicans (114 vs 111) and generally tend to stay within a standard deviation of the mean. Tweet length of Republicans are more varied and can range from very low to very high more frequently (the longest tweet is from a republican, 25 characters longer than the longest Democrat tweet). It is very likely that those with the same political affiliations may have similar vocabulary and writing style and thus produce tweets of closer length. The next feature I used was the number of ‘@’ characters, which indicate the number of other twitter users that the tweet is explicitly broadcasted to or referencing. I found that although close, Republicans use the ‘@’ symbol more frequently (~21500 vs ~21100) showing that they reference other twitter users more often, increasing the probability of a being a Republican if many references exist in the tweet. The next feature I used was the number of fully capitalized words in a tweet. I found that republicans use more fully capitalized words (~20200 vs 19000), thus once again helping differentiate republicans from democrats in this data set. Capital words can be used to express “loudness” which may possibly be a pattern in the tweets of Republicans. More data of course would be needed to confirm this. Finally, the last feature I used was the number of capital letters in the tweet since many names, landmarks, etc. start with capitalization, this feature may glean insights on what is spoken about and those of the same political affiliation may speak of similar things/topics. It can also subtly capture anger/excitement/urgency. Once again, Republicans seem to use more capital characters (~2273000 vs ~ 221000). Having these differences helps search for subtle differences between the tweets that may not be obvious from just the words used.

* I created many other features such as the number of exclamation points, number of hashtags, number of question marks, and if the tweet was a retweet. These didn’t seem to work because the values across both parties was practically identical, i.e the total number of exclamation points used by republicans in the training set was 2396 and 2399 for democrats.