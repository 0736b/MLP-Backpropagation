from random import randrange as rr

def crossvalidation(data, fold):
    dataset_split = []
    dataset_copy = list(data)
    fold_size = int(len(data) / fold)
    for i in range(fold):
        fold = list()
        while len(fold) < fold_size:
            index = rr(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split