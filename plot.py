from model.template import *
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Setup
    random.seed(630610736)
    flood_max_epoch = 1000
    cross_max_epoch = 20
    idx_epoch = list(range(1, flood_max_epoch + 1))
    idx_fold = list(range(1, 11))
    
    # res_trained_floodno1, res_tested_floodno1, min_idx_floodno1, avg_fold_error_floodno1 = flood([8,4,1], [1,2,1], 0.01, 0.01, flood_max_epoch)
    # res_trained_floodno2, res_tested_floodno2, min_idx_floodno2, avg_fold_error_floodno2 = flood([8,4,1], [1,2,1], 0.02, 0.05, flood_max_epoch)
    # res_trained_floodno3, res_tested_floodno3, min_idx_floodno3, avg_fold_error_floodno3 = flood([8,8,1], [1,2,1], 0.01, 0.02, flood_max_epoch)
    # res_trained_floodno4, res_tested_floodno4, min_idx_floodno4, avg_fold_error_floodno4 = flood([8,2,2,1], [1,2,2,1], 0.01, 0.01, flood_max_epoch)
    
    res_trained_crossno1, res_tested_crossno1, max_idx_crossno1, avg_fold_acc_crossno1 = cross([2,32,2], [1,5,1], 0.01, 0.01, cross_max_epoch)
    # for cm in res_trained_crossno1[max_idx_crossno1]:
    #     print(cm.get_accuracy())
        
        
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    class_output = ['[1,0]', '[0,1]']   
    params = {
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'axes.titleweight':'bold',
    'figure.titlesize': 'large'
    }
    #print(plt.rcParams.keys())
    plt.rcParams.update(params)
    plt.figure(figsize = (20,10))
    for i in range(0, 10,1):
        cfm = res_tested_crossno1[i].get()
        plt.subplot(2,5,i+1)
        sns.heatmap(cfm, annot=True, yticklabels=class_output, xticklabels=class_output, cmap='YlOrBr')
        plt.xlabel('Predicted',fontweight='bold')
        plt.ylabel('Actual',fontweight='bold')
        on_fold = str(i+1)
        acc = str(round(res_tested_crossno1[i].get_accuracy(), 2))
        plt.suptitle('Confusion Matrix (Tested)', fontweight='bold', fontsize=24)
        plt.title('Fold '+ on_fold + ' Accuracy: ' + acc , fontweight='bold')
        
    plt.subplots_adjust(left=0.06,bottom=0.2,right=0.97,top=0.852,wspace=0.29,hspace=0.51)
    plt.show()
        
    
    # cf_matrix = res_tested_crossno1[max_idx_crossno1].get()

    # sns.heatmap(cf_matrix, annot=True, yticklabels=class_output, xticklabels=class_output, cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Fold 1 (Tested)')
    # plt.show()
        
    # df_trained = pd.DataFrame(res_trained_crossno1[max_idx_crossno1], index=idx_epoch, columns=['Accuracy'])
    # df_trained.index.name = 'Epoch'
    
    # create Dataframe with Pandas
    # df_trained = pd.DataFrame(result_train[min_idx],index=idx_epoch, columns=['Error'])
    # df_trained.index.name = 'Epoch'
    # df_tested = pd.DataFrame(result_test,index=idx_fold, columns=['Error'])
    # df_tested.index.name = 'Fold'
    # print(df_trained)
    # print(df_tested)
    
    
    
