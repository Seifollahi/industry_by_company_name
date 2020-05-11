import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


url = "https://github.com/Seifollahi/industry_by_company_name/blob/master/data/training-testing/train-test.csv?raw=true"

df = pd.read_csv(url)

df_selected = df[["bus_name","naics_6"]]

X, Y = df_selected.bus_name, df_selected.naics_6.astype(str)
features = X.to_list()

"""
Lemmatizing the data
In this section we used Textblob mocule to lemmatize the input text.
"""
def split_into_lemmas(features):
    # features = str.encode(features, 'utf8', errors='replace').lower()
    words = TextBlob(features).words 
    return [word.lemma for word in words]

bow = CountVectorizer(analyzer=split_into_lemmas).fit(features)
print ("Length of Vocabulary : "+str(len(bow.vocabulary_)))

"""
Term Frequency times inverse document frequency (TF-IDF):
TF-IDF used to reduce the weight of most common words such as "the", "a", "an".
"""
bow_list = bow.transform(features)
tfidf_transformer = TfidfTransformer().fit(bow_list)
bow_tfidf = tfidf_transformer.transform(bow_list)
print ("Dimension of the Document-Term matrix : "+str(bow_tfidf.shape))


"""
Machine Learning
SkLearn module used to create the model and predict the labels based on the features.
"""

train, test, label_train, label_test = train_test_split(X,Y, test_size=0.1)

print ("Number of samples in Training Dataset : "+str(len(train)))
print ("Number of samples in Testing Dataset : "+str(len(test)))

from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas)),
                      ('tfidf', TfidfTransformer()),
                      ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

pipeline = pipeline.fit(train, label_train)
predicted = pipeline.predict(test)
print ("Accuracy Score SGDClassifier : "+str(accuracy_score(label_test, predicted)))


# Exporting the model to a pickle file


import pickle
with open('industry_by_company_name', 'wb') as picklefile:
    pickle.dump(pipeline,picklefile)

