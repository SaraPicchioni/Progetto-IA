# Progetto Intelligenza Artificiale

Il codice presente ha l'obiettivo di confrontare le implementazioni di Bernoulli e del Perceptron nella classificazione di documenti testuali. I due datasets si possono reperire ai seguenti link: http://qwone.com/~jason/20Newsgroups/ (è utilizzata la versione da 18828 documenti) e http://www.daviddlewis.com/resources/testcollections/reuters21578/. E' importante mantenere solo la cartella "20news-19997" del primo dataset ed eliminare l'estensione.tar dal nome della cartella del secondo dataset.
Una volta scaricati i due dataset sarà possibile eseguire i due file pyhton separatamente. Il risultato sarà la comparsa di due grafici per ogni file, rappresentanti le learning curves dell'apprendimento delle due implementazioni (durante l'esecuzione del Perceptron appare in console un'avvertenza dovuta a metodi interni: essa non andrà a compromettere il risultato; dopo alcuni secondi appariranno i due grafici).

L'algoritmo per la creazione delle learning curves è ottenuto dal sito della liberia scikit-learn: https://scikitlearn.
org/stable/auto_examples/model_selection/plot_learning_curve.html. Mentre alcuni dei metodi necessari per la modifica dei file html presenti nel secondo dataset sono reperibili sul sito https://www.quantstart.com/articles/Supervised-Learning-for-Document-Classification-with-Scikit-Learn. 

