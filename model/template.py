from time import sleep
from mlp.MultilayerPerceptron import MultilayerPerceptron as MLP
from utils.datareader import get_datafornorm, standardization, de_standardization, get_datarow
from mlp.ActivationFunc import get_fn_name
from utils.crossvalidation import crossvalidation_10
from utils.confusionmatrix import ConfusionMatrix
from math import sqrt

# MLP template for Flood
def flood(layer_list, actFn_list, momentum, learning_rate, max_epoch):
    print('------------- Multi-layer Perceptron | Specs -------------')
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
    norm_list, d_notuse, o_notuse = standardization(datas,m,s)
    # Crossvalidation 10 folds
    flood_dataset = crossvalidation_10(norm_list, 'Flood_dataset.txt')
    all_folds_error = []
    sum_error = 0.0
    # 10 Folds
    for i in range(0,10,1):
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
        for j in range(0, max_epoch, 1):
            for k in range(0, length_train, 1):
                model_training_error = mlp.train(data_inputs[k], output_labels[k], k, length_train, j)
            if j == 0 or j == ((max_epoch / 2) - 1) or j == max_epoch - 1:
                print(' @Epoch:', j+1, 'Error (on standardized data):', model_training_error)
        # Testing with
        test_with_group = flood_dataset[i]['test_with_group']
        print('[+] Testing with data group:', test_with_group)
        test_data_inputs = flood_dataset[i]['test_data_inputs']
        test_output_labels = flood_dataset[i]['test_output_labels']
        length_test = len(test_data_inputs)
        diff_square = 0.0
        sleep(1)
        print(' Result in de-standardized')
        for z in range(0, length_test, 1):
            n = z+1
            de_desired_output = round(de_standardization(test_output_labels[z][0],m,s))
            predicted_output = mlp.test(test_data_inputs[z])
            show_inputs = test_data_inputs[z].copy()
            show_inputs = [round(de_standardization(i,m,s)) for i in show_inputs]
            de_predicted_output = round(de_standardization(predicted_output[0], m, s))
            diff_square += pow((de_desired_output - de_predicted_output), 2)
            print('- Input:',show_inputs,'| Desired output:', de_desired_output, '| Predicted output:', de_predicted_output, '| Error:', de_desired_output - de_predicted_output)
        error = sqrt((1.0/n) * diff_square)
        print('Fold :', i+1, 'Root Mean Square Error:', error)
        all_folds_error.append(error)
        sum_error += error
        print('-----------------------------------------------------')
        sleep(1.5)
    print('Average Root Mean Square Error:', (sum_error) / 10)
    print('Lowest Root Mean Square Error:', min(all_folds_error), 'on Fold:', all_folds_error.index(min(all_folds_error)) + 1)

# MLP template for Cross       
def cross(layer_list, actFn_list, momentum, learning_rate, max_epoch):
    print('------------- Multi-layer Perceptron | Specs -------------')
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
    all_accuracy = []
    for i in range(0, 10,1):
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
        model_training_error = 0.0
        for j in range(0, max_epoch, 1):
            for k in range(0, length_train, 1):
                model_training_error = mlp.train(data_inputs[k], output_labels[k], k, length_train, j)
            if j == 0 or j == ((max_epoch / 2) - 1) or j == max_epoch - 1:
                print(' @Epoch:', j+1, 'Error:', model_training_error)
        # Testing with
        test_with_group = cross_dataset[i]['test_with_group']
        print('[+] Testing with data group:', test_with_group)
        test_data_inputs = cross_dataset[i]['test_data_inputs']
        test_output_labels = cross_dataset[i]['test_output_labels']
        length_test = len(test_data_inputs)
        cm = ConfusionMatrix([[1,0], [0,1]])
        sleep(1)
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
            # print(desired_output)
            # print(r_predicted)
            if (desired_output == r_predicted):
                symbol = true_symbol
            else:
                symbol = false_symbol
            print('- Input:',show_inputs,'| Desired output:', desired_output, '| Predicted output:', r_predicted, 'is', symbol)
            cm.add_data(desired_output, r_predicted)
        # print('Fold :', i+1, 'Root Mean Square Error:', error)
        # all_folds_error.append(error)
        print('\n','  Result (Confusion-Matrix)','\n')
        cm.calc_column()
        cm.print()
        sum_accuracy += cm.get_accuracy()
        all_accuracy.append(cm.get_accuracy())
        print('\n','  Fold:',i+1,'| Accuracy:',cm.get_accuracy())
        print('-----------------------------------------------------')
        sleep(1.5)
    print('Average Accuracy:', (sum_accuracy) / 10)
    print('Max Accuracy:', max(all_accuracy), 'on Fold:', all_accuracy.index(max(all_accuracy)) + 1)
        
        