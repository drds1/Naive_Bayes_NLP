import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
#load train data
df = pd.read_csv('./nlp_data/train.csv')
X, y = list(df['text']), list(df['target'])



#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#compute the bag of words on the train data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape



''' 
Transform the sparse matrix by normalising for different doccument lengths and 
occurences

Occurrence count is a good start but there is an issue: longer documents will have higher 
average count values than shorter documents, even though they might talk about the same topics.

To avoid these potential discrepancies it suffices to divide the number of 
occurrences of each word in a document by the total number of words in 
the document: these new features are called tf for Term Frequencies.

Another refinement on top of tf is to downscale weights for words that occur 
in many documents in the corpus and are therefore less informative than those 
that occur only in a smaller portion of the corpus (e.g words like "the", "and" etc)
'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



'''
Train the classifier

'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)



'''
Tie the bag of words, tfidf transformer and prediction steps together using pipeline
'''
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)

'''
analyse performance
'''
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print('ROC curve AUC = '+str(auc))


