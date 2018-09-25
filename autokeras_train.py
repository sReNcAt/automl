from keras.datasets import cifar10
from autokeras import ImageClassifier


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ImageClassifier(verbose=True, path='auto-keras/', searcher_args={'trainer_args':{'max_iter_num':5}})
    clf.fit(x_train, y_train, time_limit = 12 * 60 * 60)
    results = clf.predict(x_test)
    print (results)
