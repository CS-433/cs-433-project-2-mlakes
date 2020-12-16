import pickle
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from src.data_loading import seed_everything, split_data
from src.embeddings import create_tf_idf

MODEL_FOLDER = '../../models/'
EMBEDDINGS_FOLDER = '../../data/embeddings/'
seed_everything()

models = {
    'nb': 'Naive Bayes',
    'lg': 'Logistic Regression',
    'rf': 'Random Forest',
    'svm': 'SVM'
}


def run_tfidf_ml_model(tweets, labels, model, n_gram=1,
                       save_model=False, cross_validation=True):
    """
    Train a logistic regression model with tf-idf features.

    :param tweets: pd.DataFrame with the features
    :param labels: pd.DataFrame with the labels
    :param model: str, the machine learning model that should be fitted tp the data
    :param n_gram: int, the max number of n-gram features
    :param save_model: bool
    :param cross_validation: bool
    """
    print("\n" + "-" * 100)
    print("MODEL TO RUN: {} with tf-idf features".format(models[model]))
    print("-" * 100 + "\n")

    print("Creating tf-idf features...")
    tfidf_data = create_tf_idf(tweets['tweet'], n_gram)

    # Define classifier
    if model in ['lg']:
        clf = LogisticRegression(random_state=0)
    elif model in ['svm']:
        clf = LinearSVC()
    elif model in ['nb']:
        clf = MultinomialNB()
    elif model in ['rf']:
        clf = RandomForestClassifier()
    else:
        raise NotImplementedError('Please select a valid model type.')

    print("Fitting model...")
    if cross_validation:
        cv = KFold(n_splits=10)
        train_x = tfidf_data.to_numpy()
        labels = labels['polarity'].to_numpy()
        scores = cross_validate(clf, train_x, labels,
                                cv=cv,
                                scoring='accuracy',
                                verbose=True,
                                n_jobs=-1)
        print('Mean accuracy: {:.3f}'.format(np.mean(scores['test_score'])))
        return {'Model': model, 'Accuracy': np.mean(scores['test_score'])}

    else:
        train_x, val_x = split_data(tfidf_data, 0.9)
        train_y, val_y = split_data(labels, 0.9)
        # fit model
        clf.fit(train_x, train_y['polarity'])
        # predict
        print('Accuracy: {:.3f}'.format(clf.score(val_x, val_y['polarity'])))
        val_y_pred = clf.predict(val_x)
        print(classification_report(y_true=val_y['polarity'], y_pred=val_y_pred))

        # save model
        if save_model:
            print("Saving model...")
            pickle.dumps(clf, open(os.path.join(MODEL_FOLDER, 'finalized_model.pkl'), 'wb'))
            print("Done!")
