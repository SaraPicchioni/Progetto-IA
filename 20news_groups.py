import os
import codecs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from LearningCurves import Bern_Perc

# Path of the dataset
news_groups_folder = './20news-18828'


def clean_text(text):
    # Converte in lettere minuscole
    lower = text.lower()
    return lower


def scan_directory(folder, articles, id, folderName=""):
    # Estrazione del testo dai documenti
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            category = folderName
            with codecs.open(path, 'r', encoding='utf-8', errors='replace') as content_file:
                raw_text = content_file.read()
            text = clean_text(raw_text)
            id += 1
            articles.append((id, category, text))
        else:
            scan_directory(path, articles, id, name)


def get_list(matrix, i):
    # restitisce la matrice come una lista
    return [row[i] for row in matrix]


def create_training_data(train):
    # Crea la lista delle categorie
    y = get_list(train, 1)

    # Crea la lista del documento
    corpus = get_list(train, 2)

    # Creazione BAG OF WORDS
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    print('Vectorizer shape: ' + str(X.shape))
    return X, y


articles_data = []
scan_directory(news_groups_folder, articles_data, id=0)
X, y = create_training_data(articles_data)


# Creazione learning curves di Bernoulli e Perceptron
Bern_Perc(X,y, name="20 newsgroup")


