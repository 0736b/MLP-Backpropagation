from random import randint
from utils.datareader import get_inputoutputfromrow

def shuffle_dataset(datalist):
    shuffled = datalist.copy()
    n = len(datalist) - 1
    for i in range(n):
        rand_index = randint(0, n)
        temp = shuffled.pop(rand_index)
        shuffled.append(temp)
    return shuffled

def split(data, group_size):
    for i in range(0, len(data), group_size):
        yield data[i:i + group_size]

def crossvalidation_10(data, filename):
    # Shuffle dataset
    shuffled = shuffle_dataset(data)
    # Split into 10 groups
    group_size = int(len(shuffled) / 10)
    grouped = shuffled
    grouped = list(split(grouped, group_size))
    num_of_set = [1,2,3,4,5,6,7,8,9,10]
    if len(grouped) > 10:
        del grouped[10]
    # Creating test set
    all_testset = []
    for i in range(0, len(grouped), 1):
        t_grouped = grouped.copy()
        t_num_of_set = num_of_set.copy()
        testset = {'iterate': i+1,'train_data_inputs': [],'train_output_labels': [],'test_data_inputs': [],'test_output_labels': [],'train_with_group': [],'test_with_group': i+1}
        for j in range(0, len(grouped[i]), 1):
            input_test, output_test = get_inputoutputfromrow((grouped[i][j]), filename)
            testset['test_data_inputs'].append(input_test)
            testset['test_output_labels'].append(output_test)
        del t_grouped[i]
        del t_num_of_set[i]
        for k in range(0, len(t_grouped), 1):
            for j in range(0, len(t_grouped[k]),1):
                input_train, output_train = get_inputoutputfromrow(t_grouped[k][j], filename)
                testset['train_data_inputs'].append(input_train)
                testset['train_output_labels'].append(output_train)
                testset['train_with_group'] = t_num_of_set
        all_testset.append(testset)
    return all_testset