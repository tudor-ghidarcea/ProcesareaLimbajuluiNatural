import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# incarcarea setului de date
data = []
with open('news_category_dataset_v3.json') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# alegem 5 categorii dorite
df = df[df['category'].isin(['ENTERTAINMENT', 'POLITICS', 'WORLD NEWS', 'SPORTS', 'WEIRD NEWS'])]

# functia de preprocesare
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    stems = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(stems)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# preprocesarea titlurilor si descrierilor
df['text'] = df['headline'] + ' ' + df['short_description']
df['text'] = df['text'].apply(preprocess)

# impartim setul de date in train si test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# vectorizatori
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

# cream reprezentarile bag-of-words
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# antrenam si evaluam modelele utilizand naive bayes
nb_count = MultinomialNB()
nb_tfidf = MultinomialNB()

nb_count.fit(X_train_count, y_train)
nb_tfidf.fit(X_train_tfidf, y_train)

y_pred_count = nb_count.predict(X_test_count)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

print("Count Vectorizer - Stop words removal + Stemming + Bag-of-Words")
print("Accuracy:", accuracy_score(y_test, y_pred_count))
print("Precision:", precision_score(y_test, y_pred_count, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_count, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_count, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_count))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_count))

print("Tf-Idf Vectorizer - Stop words removal + Lemmatization + TF-IDF")

