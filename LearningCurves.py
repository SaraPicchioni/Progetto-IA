import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit


# Creazione learning curves
def plot_learning_curve_with_score(estimator, title, X, y, cv, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="b")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
                label="Cross-validation score")

        plt.legend(loc="best")
        return plt


def Bern_Perc(X,y, name):
        # Cross validation
        cv = ShuffleSplit()

        # Bernoulli learning curve
        print("Plotting Bernoulli learning curve...")
        title = "Learning Curves (Bernoulli), " + name
        plot_learning_curve_with_score(BernoulliNB(alpha=.01), title, X, y, cv)
        print("Bernoulli finished")

        # Perceptron learning curve
        print("Plotting Perceptron learning curve...")
        title = "Learning Curves (Perceptron), " + name
        plot_learning_curve_with_score(Perceptron(alpha=.01, max_iter=5, tol=None), title, X, y, cv)
        print("Perceptron finished")

        plt.show()


