from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(n_classes=9, n_labels=2, allow_unlabeled=False, random_state=42)

print(y)