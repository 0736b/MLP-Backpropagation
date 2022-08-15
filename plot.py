from model.template import *
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt

def plt_cross_cf_matrix(result, color, mode):
    res_trained_last_epoch = []
    mode_text = ''
    if mode == 'trained':
        for i in range(0, 10,1):
            res_trained_last_epoch.append(result[i][len(result[i]) - 1])
        result = res_trained_last_epoch
        mode_text = 'Trained'
    elif mode == 'tested':
        mode_text = 'Tested'
        
    class_output = ['[1,0]', '[0,1]']   
    params = {
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'axes.titleweight':'bold',
    'figure.titlesize': 'large'
    }
    
    plt.rcParams.update(params)
    plt.figure(figsize = (20,10))
    
    for i in range(0, 10,1):
        cfm = result[i].get()
        plt.subplot(2,5,i+1)
        sns.heatmap(cfm, annot=True, yticklabels=class_output, xticklabels=class_output, cmap=color)
        plt.xlabel('Predicted',fontweight='bold')
        plt.ylabel('Actual',fontweight='bold')
        on_fold = str(i+1)
        acc = str(round(result[i].get_accuracy(), 2))
        plt.suptitle('Confusion Matrix (' + mode_text + ')', fontweight='bold', fontsize=24)
        plt.title('Fold '+ on_fold + ' Accuracy: ' + acc , fontweight='bold')
        
    plt.subplots_adjust(left=0.06,bottom=0.2,right=0.97,top=0.852,wspace=0.29,hspace=0.51)
    plt.show()


def plt_cross_folds_accuracy(result):
    pass



if __name__ == '__main__':
    # Setup
    random.seed(630610736)
    flood_max_epoch = 1000
    cross_max_epoch = 2000
    idx_epoch_flood = list(range(1, flood_max_epoch + 1))
    idx_epoch_cross = list(range(1, cross_max_epoch + 1))
    idx_fold = list(range(1, 11))
    
    # res_trained_floodno1, res_tested_floodno1, min_idx_floodno1, avg_fold_error_floodno1 = flood([8,4,1], [1,2,1], 0.01, 0.01, flood_max_epoch)
    # res_trained_floodno2, res_tested_floodno2, min_idx_floodno2, avg_fold_error_floodno2 = flood([8,4,1], [1,2,1], 0.02, 0.05, flood_max_epoch)
    # res_trained_floodno3, res_tested_floodno3, min_idx_floodno3, avg_fold_error_floodno3 = flood([8,8,1], [1,2,1], 0.01, 0.02, flood_max_epoch)
    # res_trained_floodno4, res_tested_floodno4, min_idx_floodno4, avg_fold_error_floodno4 = flood([8,2,2,1], [1,2,2,1], 0.01, 0.01, flood_max_epoch)
    
    res_trained_crossno1, res_tested_crossno1, max_idx_crossno1, avg_fold_acc_crossno1 = cross([2,32,2], [1,5,1], 0.01, 0.01, cross_max_epoch)
    res_trained_crossno2, res_tested_crossno2, max_idx_crossno2, avg_fold_acc_crossno2 = cross([2,16,2], [1,5,1], 0.01, 0.01, cross_max_epoch)
    res_trained_crossno3, res_tested_crossno3, max_idx_crossno3, avg_fold_acc_crossno3 = cross([2,8,2], [1,5,1], 0.01, 0.01, cross_max_epoch)

    # plt_cross_cf_matrix(res_trained_crossno1, 'Blues', 'trained')    
    # plt_cross_cf_matrix(res_tested_crossno1, 'YlOrBr', 'tested')
         
         
    df_trained1 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno1[max_idx_crossno1]], index=idx_epoch_cross, columns=['model 2-32-2'])
    df_trained1.index.name = 'Epoch'
    
    df_trained2 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno2[max_idx_crossno2]], index=idx_epoch_cross, columns=['model 2-16-2'])
    df_trained2.index.name = 'Epoch'
    
    df_trained3 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno3[max_idx_crossno3]], index=idx_epoch_cross, columns=['model 2-8-2'])
    df_trained3.index.name = 'Epoch'
    
    result = pd.concat([df_trained1, df_trained2, df_trained3], axis=1)
    sns.lineplot(data=result)
    plt.title('Cross Accuracy converge')
    plt.ylabel('Accuracy')
    plt.show()
    
    
    
