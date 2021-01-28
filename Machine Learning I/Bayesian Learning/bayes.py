from pprint import pprint
from time import time
import logging
import pickle
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from string import punctuation

print(__doc__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# #############################################################################
data = fetch_20newsgroups(subset='all', remove=('headers', "footers", "quotes"))

tbl = str.maketrans({ord(ch): " " for ch in punctuation})
amap = map(lambda x: x.translate(tbl).replace("\n", " "), data.data)
alist = []
for i in amap:
    c = "  ".join(i.split())
    alist.append(c)
data.data = alist
print("%d Documents" % len(data.filenames))
print("%d Categories" % len(data.target_names))
print()

# #############################################################################

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])


parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.10, 0.25, 0.50, 0.75, 1.0),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    try:
        grid_search = pickle.load(open("grid_search_cv.p", "rb"))
    except Exception:
        grid_search = GridSearchCV(pipeline, parameters, scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'],
                                   n_jobs=-1, refit='f1_macro', verbose=1)

        grid_search.fit(data.data, data.target)
        # print("done in %0.3fs" % (time() - t0))
        # print()

        pickle.dump(grid_search, open("grid_search_cv.p", "wb"))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
