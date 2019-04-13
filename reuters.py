import html
import re
from html.parser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from LearningCurves import Bern_Perc

from collections import Counter

# Path of the reuters dataset
reuters_path = "./reuters21578/"


class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.
    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    """
    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag".
        If the tag is a  tag, then we remove all
        white-space with a regular expression and then append the
        topic-body tuple.
        If the tag is a  or  tag then we simply set
        the internal state to False for these booleans, respectively.
        If the tag is a  tag (found within a  tag), then we
        append the particular topic to the "topics" list and
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


def get_topic_tags():
    # Apre il file delle categorie ed estrapola i nomi
    topics = open(reuters_path + "all-topics-strings.lc.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    return topics


def get_frequent_topic_list(topics, docs):
    # Restituisce la lista con le 10 categorie più frequenti
    topics_occurrences = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics:
                topics_occurrences.append(t)
                break
    return list(dict(Counter(topics_occurrences).most_common(10)).keys())


def topics_filter(frequent_topics, docs):
    # Legge i documenti e crea una nuova lista di due tuple contenente una categoria e il testo
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in frequent_topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs


def create_training_data(docs):
    # Crea la lista con le etichette
    y = [d[0] for d in docs]

    # Crea la lista del documento
    corpus = [d[1] for d in docs]

    # Creazione BAG OF WORDS
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    print('Vectorizer shape: ' + str(X.shape))
    return X, y


# Crea la lista dei dati
uri = reuters_path + "reut2-%03d.sgm"
files = [uri % r for r in range(0, 22)]
parser = ReutersParser()

docs = []
for fn in files:
    for d in parser.parse(open(fn, 'rb')):
        docs.append(d)


topics = get_topic_tags()
ten_most_common_topics = get_frequent_topic_list(topics, docs)
ref_docs = topics_filter(ten_most_common_topics, docs)
X, y = create_training_data(ref_docs)


# Creazione learning curves di Bernoulli e Perceptron
Bern_Perc(X,y, name="Reuters")
