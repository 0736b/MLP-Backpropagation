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


def plt_flood_err_trained(result, model):
    plt.figure(figsize = (20,10))
    for i in range(0, 10,1):
        error = result[i]
        plt.subplot(2,5,i+1)
        er_train = pd.DataFrame(error, index=idx_epoch_flood, columns=['flood model ' + model])
        er_train.index.name = 'Epoch'
        sns.lineplot(data=er_train)
        plt.ylabel('RMSE')
        on_fold = str(i+1)
        plt.suptitle('Flood ' + model + ' RMSE Converge', fontweight='bold', fontsize=24)
        plt.title('Iteration ' + on_fold + ', RMSE at last epoch: ' + str(round(error[len(error) - 1],2)), fontweight='bold')
    plt.subplots_adjust(left=0.04,bottom=0.2,right=0.97,top=0.852,wspace=0.29,hspace=0.51)
    plt.show()
    
def plt_flood_lastepoch(result_trained, result_tested, model):
    idx = []
    trained_l = []
    test_name = ['Train'] * 10
    train_name = ['Test'] * 10
    for i in range(0, 10,1):
        trained_l.append(result_trained[i][len(result_trained[i]) - 1])
        idx.append(str(i+1))
    er_train_l = pd.DataFrame(trained_l, columns=['RMSE'])
    er_train_n = pd.DataFrame(train_name, columns=['Train/Test'])
    df_idx = pd.DataFrame(idx, columns=['Iteration'])
    merge_train = pd.concat([df_idx,er_train_l, er_train_n], axis=1)
    
    er_tested_l = pd.DataFrame(result_tested, columns=['RMSE'])
    er_tested_n = pd.DataFrame(test_name, columns=['Train/Test'])
    merge_tested = pd.concat([df_idx, er_tested_l, er_tested_n], axis=1)
    
    merge_er = pd.concat([merge_train, merge_tested])
    sns.barplot(x='Iteration', y='RMSE', data=merge_er, hue='Train/Test')
    plt.title('Flood model ' + model + ' Train/Test RMSE', fontweight='bold', fontsize=24)
    plt.ylabel('RMSE')
    plt.show()



if __name__ == '__main__':
    # Setup
    random.seed(630610736)
    flood_max_epoch = 1000
    cross_max_epoch = 2000
    idx_epoch_flood = list(range(1, flood_max_epoch + 1))
    idx_epoch_cross = list(range(1, cross_max_epoch + 1))
    idx_fold = list(range(1, 11))
    
    # Flood Experiment
    res_trained_floodno1, res_tested_floodno1, min_idx_floodno1, avg_fold_error_floodno1 = flood([8,4,1], [1,2,1], 0.01, 0.01, flood_max_epoch)
    res_trained_floodno2, res_tested_floodno2, min_idx_floodno2, avg_fold_error_floodno2 = flood([8,4,1], [1,2,1], 0.02, 0.05, flood_max_epoch)
    res_trained_floodno3, res_tested_floodno3, min_idx_floodno3, avg_fold_error_floodno3 = flood([8,8,1], [1,2,1], 0.01, 0.02, flood_max_epoch)
    res_trained_floodno4, res_tested_floodno4, min_idx_floodno4, avg_fold_error_floodno4 = flood([8,2,2,1], [1,2,2,1], 0.01, 0.01, flood_max_epoch)
    
    # plt_flood_err_trained(res_trained_floodno1, '8-4-1')
    # plt_flood_err_trained(res_trained_floodno2, '8-4-1')
    # plt_flood_err_trained(res_trained_floodno3, '8-8-1')
    # plt_flood_err_trained(res_trained_floodno4, '8-2-2-1')
    
    # plt_flood_lastepoch(res_trained_floodno1, res_tested_floodno1, '8-4-1')
    # plt_flood_lastepoch(res_trained_floodno2, res_tested_floodno2, '8-4-1')
    # plt_flood_lastepoch(res_trained_floodno3, res_tested_floodno3, '8-8-1')
    # plt_flood_lastepoch(res_trained_floodno4, res_tested_floodno4, '8-2-2-1')
    
    # Comparing RMSE converge all models
    er_trained_1 = pd.DataFrame([float(er) for er in res_trained_floodno1[min_idx_floodno1]], index=idx_epoch_flood, columns=['8-4-1 LR=0.01 MR=0.01'])
    er_trained_2 = pd.DataFrame([float(er) for er in res_trained_floodno2[min_idx_floodno2]], index=idx_epoch_flood, columns=['8-4-1 LR=0.05 MR=0.02'])
    er_trained_3 = pd.DataFrame([float(er) for er in res_trained_floodno3[min_idx_floodno3]], index=idx_epoch_flood, columns=['8-8-1 LR=0.02 MR=0.01'])
    er_trained_4 = pd.DataFrame([float(er) for er in res_trained_floodno4[min_idx_floodno4]], index=idx_epoch_flood, columns=['8-2-2-1 LR=0.01 MR=0.01'])
    er_trained_1.index.name = 'Epoch'
    er_trained_2.index.name = 'Epoch'
    er_trained_3.index.name = 'Epoch'
    er_trained_4.index.name = 'Epoch'
    
    merged_er = pd.concat([er_trained_1, er_trained_2, er_trained_3, er_trained_4], axis=1)
    ax = sns.lineplot(data=merged_er)
    ax.lines[0].set_linestyle('solid')
    ax.lines[1].set_linestyle('solid')
    ax.lines[2].set_linestyle('solid')
    ax.lines[3].set_linestyle('solid')
    plt.title('Flood RMSE converge')
    plt.ylabel('RMSE')
    plt.show()
    
    # Cross Experiment
    # res_trained_crossno1, res_tested_crossno1, max_idx_crossno1, avg_fold_acc_crossno1 = cross([2,32,2], [1,5,1], 0.01, 0.01, cross_max_epoch)
    # res_trained_crossno2, res_tested_crossno2, max_idx_crossno2, avg_fold_acc_crossno2 = cross([2,4,4,2], [1,5,5,1], 0.01, 0.01, cross_max_epoch)
    # res_trained_crossno3, res_tested_crossno3, max_idx_crossno3, avg_fold_acc_crossno3 = cross([2,8,2], [1,5,1], 0.01, 0.01, cross_max_epoch)
    # res_trained_crossno4, res_tested_crossno4, max_idx_crossno4, avg_fold_acc_crossno4 = cross([2,8,2], [1,5,1], 0.01, 0.01, cross_max_epoch)

    # plt_cross_cf_matrix(res_trained_crossno1, 'Blues', 'trained')    
    # plt_cross_cf_matrix(res_tested_crossno1, 'YlOrBr', 'tested')
    
    # plt_cross_cf_matrix(res_trained_crossno2, 'Blues', 'trained')    
    # plt_cross_cf_matrix(res_tested_crossno2, 'YlOrBr', 'tested')
    
    # plt_cross_cf_matrix(res_trained_crossno3, 'Blues', 'trained')    
    # plt_cross_cf_matrix(res_tested_crossno3, 'YlOrBr', 'tested')
    
    # plt_cross_cf_matrix(res_trained_crossno4, 'Blues', 'trained')    
    # plt_cross_cf_matrix(res_tested_crossno4, 'YlOrBr', 'tested')
         
         
    # ac_trained1 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno1[max_idx_crossno1]], index=idx_epoch_cross, columns=['model 2-32-2'])
    # ac_trained1.index.name = 'Epoch'
    
    # ac_trained2 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno2[max_idx_crossno2]], index=idx_epoch_cross, columns=['model 2-16-2'])
    # ac_trained2.index.name = 'Epoch'
    
    # ac_trained3 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno3[max_idx_crossno3]], index=idx_epoch_cross, columns=['model 2-8-2'])
    # ac_trained3.index.name = 'Epoch'
    
    # ac_trained4 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno4[max_idx_crossno4]], index=idx_epoch_cross, columns=['model 2-2-2-2'])
    # ac_trained4.index.name = 'Epoch'
    
    # Comparing Accuracy converge all models
    # merged_ac = pd.concat([ac_trained1, ac_trained2, ac_trained3, ac_trained4], axis=1)
    # ax = sns.lineplot(data=merged_ac)
    # ax.lines[0].set_linestyle('solid')
    # ax.lines[1].set_linestyle('solid')
    # ax.lines[2].set_linestyle('solid')
    # ax.lines[3].set_linestyle('solid')
    # plt.title('Cross Accuracy converge')
    # plt.ylabel('Accuracy')
    # plt.show()