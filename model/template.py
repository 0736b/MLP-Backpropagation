import copy
from mlp.MultilayerPerceptron import MultilayerPerceptron as MLP
from utils.datareader import get_datafornorm, standardization, de_standardization, get_datarow
from mlp.ActivationFunc import get_fn_name
from utils.crossvalidation import crossvalidation_10
from utils.confusionmatrix import ConfusionMatrix
from math import sqrt
from random import shuffle

# MLP template for Flood
def flood(layer_list, actFn_list, momentum, learning_rate, max_epoch):
    # For plotting
    trained_for_plot = []
    # Printing spec
    print('\n------------- Multi-layer Perceptron | Specs -------------')
    print('[+] Input layer :', layer_list[0], 'Neurons')
    for i in range(1, len(layer_list)-1, 1):
        print('[+] Hidden Layer', i, ":", layer_list[i], 'Neurons','( Activation Function:', get_fn_name(actFn_list[i]), ')')
    print('[+] Output layer :', layer_list[len(layer_list) - 1], 'Neurons','( Activation Function:', get_fn_name(actFn_list[len(actFn_list) - 1]), ')')
    print('[+] Momentum rate :', momentum)
    print('[+] Learning rate :', learning_rate)
    print('[+] Max epoch :', max_epoch, '\n')
    print('-------------- Flood_dataset.txt (standardized) | Training --------------')
    # Loading Data
    datas, m, s = get_datafornorm('Flood_dataset.txt')
    # Standardize Data
    norm_list = standardization(datas,m,s)
    # Crossvalidation 10 folds
    flood_dataset = crossvalidation_10(norm_list, 'Flood_dataset.txt')
    all_folds_error = []
    sum_error = 0.0
    # 10 Folds
    for i in range(0,10,1):
        t_epoch_error = [None] * max_epoch
        print('Fold :', i+1)
        # Init MLP
        mlp = MLP(layer_list, actFn_list, momentum, learning_rate)
        train_with_group = flood_dataset[i]['train_with_group']
        print('[+] Training with data group:', train_with_group)
        # Training
        # Get training dataset from 1 Fold
        data_inputs = flood_dataset[i]['train_data_inputs']
        output_labels = flood_dataset[i]['train_output_labels']
        length_train = len(data_inputs)
        model_training_error = 0.0
        input_shuffled = list(range(0,length_train))
        for j in range(0, max_epoch, 1):
            shuffle(input_shuffled)
            t = 0
            for k in input_shuffled:
                model_training_error = mlp.train(data_inputs[k], output_labels[k], t, length_train, 'regression', None)
                t += 1
            # collecting data for plotting
            t_epoch_error[j] = (model_training_error)
            if j == 0 or j == ((max_epoch / 2) - 1) or j == max_epoch - 1:
                print(' @Epoch:', j+1, 'Error (on standardized data):', model_training_error)
        trained_for_plot.append(copy.deepcopy(t_epoch_error))
        # Testing with
        test_with_group = flood_dataset[i]['test_with_group']
        print('[+] Testing with data group:', test_with_group)
        test_data_inputs = flood_dataset[i]['test_data_inputs']
        test_output_labels = flood_dataset[i]['test_output_labels']
        length_test = len(test_data_inputs)
        diff_square = 0.0
        print(' Result in de-standardized')
        for z in range(0, length_test, 1):
            de_desired_output = round(de_standardization(test_output_labels[z][0],m,s))
            predicted_output = mlp.test(test_data_inputs[z])
            show_inputs = test_data_inputs[z].copy()
            show_inputs = [round(de_standardization(i,m,s)) for i in show_inputs]
            de_predicted_output = round(de_standardization(predicted_output[0], m, s))
            predict_error = predicted_output[0] - test_output_labels[z][0]
            diff_square += (predict_error ** 2)
            print('- Input:',show_inputs,'| Desired output:', de_desired_output, '| Predicted output:', de_predicted_output, '| Error:', de_desired_output - de_predicted_output)
        error = diff_square / float(length_test)
        error = sqrt(error)
        print('Fold :', i+1, 'Root Mean Square Error:', error)
        all_folds_error.append(error)
        sum_error += error
        print('-----------------------------------------------------')
    print('Average Root Mean Square Error:', (sum_error) / 10)
    print('Lowest Root Mean Square Error:', min(all_folds_error), 'on Fold:', all_folds_error.index(min(all_folds_error)) + 1)
    min_idx = all_folds_error.index(min(all_folds_error))
    avg_error_all_fold = (sum_error) / 10
    tested_for_plot = all_folds_error.copy()
    return trained_for_plot, tested_for_plot, min_idx, avg_error_all_fold

# MLP template for Cross     
def cross(layer_list, actFn_list, momentum, learning_rate, max_epoch):
    # For plotting
    trained_for_plot = []
    tested_for_plot = []
    # Printing spec
    print('\n------------- Multi-layer Perceptron | Specs -------------')
    print('[+] Input layer :', layer_list[0], 'Neurons')
    for i in range(1, len(layer_list)-1, 1):
        print('[+] Hidden Layer', i, ":", layer_list[i], 'Neurons','( Activation Function:', get_fn_name(actFn_list[i]), ')')
    print('[+] Output layer :', layer_list[len(layer_list) - 1], 'Neurons','( Activation Function:', get_fn_name(actFn_list[len(actFn_list) - 1]), ')')
    print('[+] Momentum rate :', momentum)
    print('[+] Learning rate :', learning_rate)
    print('[+] Max epoch :', max_epoch, '\n')
    print('-------------- cross.pat | Training --------------')
    # Loading Data
    data = get_datarow('cross.pat')
    # Crossvalidation 10 folds
    cross_dataset = crossvalidation_10(data, 'cross.pat')
    true_symbol = '✅'
    false_symbol = '❌'
    sum_accuracy = 0.0
    sum_misclass = 0.0
    all_accuracy = []
    for i in range(0, 10,1):
        t_cm_trained = [None] * max_epoch
        print('Fold :', i+1)
        # Init MLP
        mlp = MLP(layer_list, actFn_list, momentum, learning_rate)
        train_with_group = cross_dataset[i]['train_with_group']
        print('[+] Training with data group:', train_with_group)
        # Training
        # Get training dataset from 1 Fold
        data_inputs = cross_dataset[i]['train_data_inputs']
        output_labels = cross_dataset[i]['train_output_labels']
        length_train = len(data_inputs)
        model_training_acc = 0.0
        cm_train = ConfusionMatrix([[1,0], [0,1]])
        input_shuffled = list(range(0,length_train))
        for j in range(0, max_epoch, 1):
            shuffle(input_shuffled)
            t = 0
            for k in input_shuffled:
                model_training_acc = mlp.train(data_inputs[k], output_labels[k], t, length_train, 'classification', cm_train)
                t += 1
            if j == 0 or j == ((max_epoch / 2) - 1) or j == max_epoch - 1:
                print(' @Epoch:', j+1, 'Accuracy:', model_training_acc)
            t_cm_trained[j] = copy.deepcopy(cm_train)
            cm_train.clear()
        # Testing with
        trained_for_plot.append(t_cm_trained.copy())
        test_with_group = cross_dataset[i]['test_with_group']
        print('[+] Testing with data group:', test_with_group)
        test_data_inputs = cross_dataset[i]['test_data_inputs']
        test_output_labels = cross_dataset[i]['test_output_labels']
        length_test = len(test_data_inputs)
        cm_test = ConfusionMatrix([[1,0], [0,1]])
        print('  Result (raw)')
        for z in range(0, length_test, 1):
            symbol = ''
            desired_output = test_output_labels[z]
            predicted_output = mlp.test(test_data_inputs[z])
            show_inputs = test_data_inputs[z].copy()
            r_predicted = predicted_output.copy()
            if r_predicted[0] > r_predicted[1]:
                r_predicted[0] = 1
                r_predicted[1] = 0
            elif r_predicted[0] < r_predicted[1]:
                r_predicted[0] = 0
                r_predicted[1] = 1
            if (desired_output == r_predicted):
                symbol = true_symbol
            else:
                symbol = false_symbol
            print('- Input:',show_inputs,'| Desired output:', desired_output, '| Predicted output:', r_predicted, 'is', symbol)
            cm_test.add_data(desired_output, r_predicted)
        print('\n','  Result (Confusion-Matrix)','\n')
        cm_test.calc_column()
        cm_test.print()
        sum_accuracy += cm_test.get_accuracy()
        sum_misclass += cm_test.get_misclass()
        all_accuracy.append(cm_test.get_accuracy())
        tested_for_plot.append(copy.deepcopy(cm_test))
        print('\n','  Fold:',i+1,'| Accuracy:',cm_test.get_accuracy())
        print('   Fold:',i+1,'| Misclassification:',cm_test.get_misclass())
        print('-----------------------------------------------------')
    print('Average Accuracy:', (sum_accuracy) / 10)
    print('Average Misclassification:', (sum_misclass) / 10)
    print('Max Accuracy:', max(all_accuracy), 'on Fold:', all_accuracy.index(max(all_accuracy)) + 1)
    max_idx = all_accuracy.index(max(all_accuracy))
    avg_acc = (sum_accuracy) / 10
    return trained_for_plot, tested_for_plot, max_idx, avg_acc