from model.template import *
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import pickle

def plt_cross_cf_matrix(result, color, mode, model):
    res_trained_last_epoch = []
    mode_text = ''
    if mode == 'trained':
        for i in range(0, 10,1):
            res_trained_last_epoch.append(result[i][len(result[i]) - 1])
        result = res_trained_last_epoch
        mode_text = 'Training'
    elif mode == 'tested':
        mode_text = 'Validation'
        
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
        acc = str(round(result[i].get_accuracy(), 4))
        plt.suptitle('Cross ' + model +  '\nConfusion Matrix (' + mode_text + ')', fontweight='bold', fontsize=24)
        plt.title('Fold '+ on_fold + ' Accuracy: ' + acc , fontweight='bold')
        
    plt.subplots_adjust(left=0.06,bottom=0.14,right=0.97,top=0.788,wspace=0.29,hspace=0.51)
    plt.show()

def plt_flood_err_trained(result, model):
    plt.figure(figsize = (20,10))
    for i in range(0, 10,1):
        error = result[i]
        plt.subplot(2,5,i+1)
        er_train = pd.DataFrame(error, index=idx_epoch_flood, columns=[model])
        er_train.index.name = 'Epoch'
        sns.lineplot(data=er_train)
        plt.ylabel('Root Mean Square Error (RMSE)')
        on_fold = str(i+1)
        plt.suptitle('Flood ' + model + '\nRMSE Converge', fontweight='bold', fontsize=24)
        plt.title('Fold ' + on_fold + ', RMSE at last epoch: ' + str(round(error[len(error) - 1],4)), fontweight='bold')
    plt.subplots_adjust(left=0.04,bottom=0.117,right=0.97,top=0.817,wspace=0.29,hspace=0.51)
    plt.show()
    
def plt_flood_lastepoch(result_trained, result_tested, model):
    idx = []
    trained_l = []
    test_name = ['Validation'] * 10
    train_name = ['Training'] * 10
    for i in range(0, 10,1):
        trained_l.append(result_trained[i][len(result_trained[i]) - 1])
        idx.append(str(i+1))
    er_train_l = pd.DataFrame(trained_l, columns=['RMSE'])
    er_train_n = pd.DataFrame(train_name, columns=['Training/Validation'])
    df_idx = pd.DataFrame(idx, columns=['Iteration'])
    merge_train = pd.concat([df_idx,er_train_l, er_train_n], axis=1)
    
    er_tested_l = pd.DataFrame(result_tested, columns=['RMSE'])
    er_tested_n = pd.DataFrame(test_name, columns=['Training/Validation'])
    merge_tested = pd.concat([df_idx, er_tested_l, er_tested_n], axis=1)
    
    merge_er = pd.concat([merge_train, merge_tested])
    sns.barplot(x='Iteration', y='RMSE', data=merge_er, hue='Training/Validation')
    plt.title('Flood model ' + model + '\nTraining/Validation RMSE', fontweight='bold', fontsize=24)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=18)
    plt.xlabel('Fold', fontsize=18)
    plt.show()

def flood_runsave(flood_max_epoch):
    # Flood Experiment
    random.seed(1)
    max_epoch = flood_max_epoch
    
    res_trained_floodno1, res_tested_floodno1, min_idx_floodno1, avg_fold_error_floodno1 = flood([8,4,1], [1,2,1], 0.01, 0.01, max_epoch)
    with open("saved/flood/res_trained_floodno1.data", "wb") as fp:
        pickle.dump(res_trained_floodno1, fp)
    with open("saved/flood/res_tested_floodno1.data", "wb") as fp:
        pickle.dump(res_tested_floodno1, fp)
    with open("saved/flood/min_idx_floodno1.data", "wb") as fp:
        pickle.dump(min_idx_floodno1, fp)
    with open("saved/flood/avg_fold_error_floodno1.data", "wb") as fp:
        pickle.dump(avg_fold_error_floodno1, fp)

    res_trained_floodno2, res_tested_floodno2, min_idx_floodno2, avg_fold_error_floodno2 = flood([8,4,1], [1,2,1], 0.03, 0.05, max_epoch)
    with open("saved/flood/res_trained_floodno2.data", "wb") as fp:
        pickle.dump(res_trained_floodno2, fp)
    with open("saved/flood/res_tested_floodno2.data", "wb") as fp:
        pickle.dump(res_tested_floodno2, fp)
    with open("saved/flood/min_idx_floodno2.data", "wb") as fp:
        pickle.dump(min_idx_floodno2, fp)
    with open("saved/flood/avg_fold_error_floodno2.data", "wb") as fp:
        pickle.dump(avg_fold_error_floodno2, fp)
    
    res_trained_floodno3, res_tested_floodno3, min_idx_floodno3, avg_fold_error_floodno3 = flood([8,8,1], [1,2,1], 0.01, 0.01, max_epoch)
    with open("saved/flood/res_trained_floodno3.data", "wb") as fp:
        pickle.dump(res_trained_floodno3, fp)
    with open("saved/flood/res_tested_floodno3.data", "wb") as fp:
        pickle.dump(res_tested_floodno3, fp)
    with open("saved/flood/min_idx_floodno3.data", "wb") as fp:
        pickle.dump(min_idx_floodno3, fp)
    with open("saved/flood/avg_fold_error_floodno3.data", "wb") as fp:
        pickle.dump(avg_fold_error_floodno3, fp)
    
    res_trained_floodno4, res_tested_floodno4, min_idx_floodno4, avg_fold_error_floodno4 = flood([8,2,2,1], [1,2,2,1], 0.01, 0.01, max_epoch)
    with open("saved/flood/res_trained_floodno4.data", "wb") as fp:
        pickle.dump(res_trained_floodno4, fp)
    with open("saved/flood/res_tested_floodno4.data", "wb") as fp:
        pickle.dump(res_tested_floodno4, fp)
    with open("saved/flood/min_idx_floodno4.data", "wb") as fp:
        pickle.dump(min_idx_floodno4, fp)
    with open("saved/flood/avg_fold_error_floodno4.data", "wb") as fp:
        pickle.dump(avg_fold_error_floodno4, fp)
    
def cross_runsave(cross_max_epoch):
    # Cross Experiment
    random.seed(1)
    max_epoch = cross_max_epoch
    
    res_trained_crossno1, res_tested_crossno1, max_idx_crossno1, avg_fold_acc_crossno1 = cross([2,4,2], [1,4,1], 0.01, 0.01, max_epoch)
    with open("saved/cross/res_trained_crossno1.data", "wb") as fp:
        pickle.dump(res_trained_crossno1, fp)
    with open("saved/cross/res_tested_crossno1.data", "wb") as fp:
        pickle.dump(res_tested_crossno1, fp)
    with open("saved/cross/max_idx_crossno1.data", "wb") as fp:
        pickle.dump(max_idx_crossno1, fp)
    with open("saved/cross/avg_fold_acc_crossno1.data", "wb") as fp:
        pickle.dump(avg_fold_acc_crossno1, fp)
        
    res_trained_crossno2, res_tested_crossno2, max_idx_crossno2, avg_fold_acc_crossno2 = cross([2,4,2], [1,4,1], 0.06, 0.005, max_epoch)
    with open("saved/cross/res_trained_crossno2.data", "wb") as fp:
        pickle.dump(res_trained_crossno2, fp)
    with open("saved/cross/res_tested_crossno2.data", "wb") as fp:
        pickle.dump(res_tested_crossno2, fp)
    with open("saved/cross/max_idx_crossno2.data", "wb") as fp:
        pickle.dump(max_idx_crossno2, fp)
    with open("saved/cross/avg_fold_acc_crossno2.data", "wb") as fp:
        pickle.dump(avg_fold_acc_crossno2, fp)
    
    res_trained_crossno3, res_tested_crossno3, max_idx_crossno3, avg_fold_acc_crossno3 = cross([2,8,2], [1,4,1], 0.01, 0.01, max_epoch)
    with open("saved/cross/res_trained_crossno3.data", "wb") as fp:
        pickle.dump(res_trained_crossno3, fp)
    with open("saved/cross/res_tested_crossno3.data", "wb") as fp:
        pickle.dump(res_tested_crossno3, fp)
    with open("saved/cross/max_idx_crossno3.data", "wb") as fp:
        pickle.dump(max_idx_crossno3, fp)
    with open("saved/cross/avg_fold_acc_crossno3.data", "wb") as fp:
        pickle.dump(avg_fold_acc_crossno3, fp)
    
    res_trained_crossno4, res_tested_crossno4, max_idx_crossno4, avg_fold_acc_crossno4 = cross([2,4,4,2], [1,4,4,1], 0.01, 0.01, max_epoch)
    with open("saved/cross/res_trained_crossno4.data", "wb") as fp:
        pickle.dump(res_trained_crossno4, fp)
    with open("saved/cross/res_tested_crossno4.data", "wb") as fp:
        pickle.dump(res_tested_crossno4, fp)     
    with open("saved/cross/max_idx_crossno4.data", "wb") as fp:
        pickle.dump(max_idx_crossno4, fp)
    with open("saved/cross/avg_fold_acc_crossno4.data", "wb") as fp:
        pickle.dump(avg_fold_acc_crossno4, fp)
    
def flood_loadresult():
    with open("saved/flood/res_trained_floodno1.data", "rb") as fp:
        res_trained_floodno1 = pickle.load(fp)
    with open("saved/flood/res_trained_floodno2.data", "rb") as fp:
        res_trained_floodno2 = pickle.load(fp)
    with open("saved/flood/res_trained_floodno3.data", "rb") as fp:
        res_trained_floodno3 = pickle.load(fp)
    with open("saved/flood/res_trained_floodno4.data", "rb") as fp:
        res_trained_floodno4 = pickle.load(fp)
    
    with open("saved/flood/res_tested_floodno1.data", "rb") as fp:
        res_tested_floodno1 = pickle.load(fp)
    with open("saved/flood/res_tested_floodno2.data", "rb") as fp:
        res_tested_floodno2 = pickle.load(fp)
    with open("saved/flood/res_tested_floodno3.data", "rb") as fp:
        res_tested_floodno3 = pickle.load(fp)
    with open("saved/flood/res_tested_floodno4.data", "rb") as fp:
        res_tested_floodno4 = pickle.load(fp)
        
    with open("saved/flood/min_idx_floodno1.data", "rb") as fp:
        min_idx_floodno1 = pickle.load(fp)
    with open("saved/flood/min_idx_floodno2.data", "rb") as fp:
        min_idx_floodno2 = pickle.load(fp)
    with open("saved/flood/min_idx_floodno3.data", "rb") as fp:
        min_idx_floodno3 = pickle.load(fp)
    with open("saved/flood/min_idx_floodno4.data", "rb") as fp:
        min_idx_floodno4 = pickle.load(fp)
        
    with open("saved/flood/avg_fold_error_floodno1.data", "rb") as fp:
        avg_fold_error_floodno1 = pickle.load(fp)
    with open("saved/flood/avg_fold_error_floodno2.data", "rb") as fp:
        avg_fold_error_floodno2 = pickle.load(fp)
    with open("saved/flood/avg_fold_error_floodno3.data", "rb") as fp:
        avg_fold_error_floodno3 = pickle.load(fp)
    with open("saved/flood/avg_fold_error_floodno4.data", "rb") as fp:
        avg_fold_error_floodno4 = pickle.load(fp)
        
    return res_trained_floodno1,res_trained_floodno2,res_trained_floodno3,res_trained_floodno4,res_tested_floodno1,res_tested_floodno2,res_tested_floodno3,res_tested_floodno4,min_idx_floodno1,min_idx_floodno2,min_idx_floodno3,min_idx_floodno4,avg_fold_error_floodno1,avg_fold_error_floodno2,avg_fold_error_floodno3,avg_fold_error_floodno4
    
def cross_loadresult():
    with open("saved/cross/res_trained_crossno1.data", "rb") as fp:
        res_trained_crossno1 = pickle.load(fp)
    with open("saved/cross/res_trained_crossno2.data", "rb") as fp:
        res_trained_crossno2 = pickle.load(fp)
    with open("saved/cross/res_trained_crossno3.data", "rb") as fp:
        res_trained_crossno3 = pickle.load(fp)
    with open("saved/cross/res_trained_crossno4.data", "rb") as fp:
        res_trained_crossno4 = pickle.load(fp)
        
    with open("saved/cross/res_tested_crossno1.data", "rb") as fp:
        res_tested_crossno1 = pickle.load(fp)
    with open("saved/cross/res_tested_crossno2.data", "rb") as fp:
        res_tested_crossno2 = pickle.load(fp)
    with open("saved/cross/res_tested_crossno3.data", "rb") as fp:
        res_tested_crossno3 = pickle.load(fp)
    with open("saved/cross/res_tested_crossno4.data", "rb") as fp:
        res_tested_crossno4 = pickle.load(fp)
        
    with open("saved/cross/max_idx_crossno1.data", "rb") as fp:
        max_idx_crossno1 = pickle.load(fp)
    with open("saved/cross/max_idx_crossno2.data", "rb") as fp:
        max_idx_crossno2 = pickle.load(fp)
    with open("saved/cross/max_idx_crossno3.data", "rb") as fp:
        max_idx_crossno3 = pickle.load(fp)
    with open("saved/cross/max_idx_crossno4.data", "rb") as fp:
        max_idx_crossno4 = pickle.load(fp)
        
    with open("saved/cross/avg_fold_acc_crossno1.data", "rb") as fp:
        avg_fold_acc_crossno1 = pickle.load(fp)
    with open("saved/cross/avg_fold_acc_crossno2.data", "rb") as fp:
        avg_fold_acc_crossno2 = pickle.load(fp)
    with open("saved/cross/avg_fold_acc_crossno3.data", "rb") as fp:
        avg_fold_acc_crossno3 = pickle.load(fp)
    with open("saved/cross/avg_fold_acc_crossno4.data", "rb") as fp:
        avg_fold_acc_crossno4 = pickle.load(fp)
    
    return res_trained_crossno1,res_trained_crossno2,res_trained_crossno3,res_trained_crossno4,res_tested_crossno1,res_tested_crossno2,res_tested_crossno3,res_tested_crossno4,max_idx_crossno1,max_idx_crossno2,max_idx_crossno3,max_idx_crossno4,avg_fold_acc_crossno1,avg_fold_acc_crossno2,avg_fold_acc_crossno3,avg_fold_acc_crossno4

if __name__ == '__main__':
    # Setup
    flood_max_epoch = 2000
    cross_max_epoch = 2500
    idx_epoch_flood = list(range(1, flood_max_epoch + 1))
    idx_epoch_cross = list(range(1, cross_max_epoch + 1))
    idx_fold = list(range(1, 11))
    
    # Running MLP and save result locally
    # cross_runsave(cross_max_epoch)
    # flood_runsave(flood_max_epoch)
    
    # Load result
    res_trained_floodno1,res_trained_floodno2,res_trained_floodno3,res_trained_floodno4,res_tested_floodno1,res_tested_floodno2,res_tested_floodno3,res_tested_floodno4,min_idx_floodno1,min_idx_floodno2,min_idx_floodno3,min_idx_floodno4,avg_fold_error_floodno1,avg_fold_error_floodno2,avg_fold_error_floodno3,avg_fold_error_floodno4 = flood_loadresult()
    res_trained_crossno1,res_trained_crossno2,res_trained_crossno3,res_trained_crossno4,res_tested_crossno1,res_tested_crossno2,res_tested_crossno3,res_tested_crossno4,max_idx_crossno1,max_idx_crossno2,max_idx_crossno3,max_idx_crossno4,avg_fold_acc_crossno1,avg_fold_acc_crossno2,avg_fold_acc_crossno3,avg_fold_acc_crossno4 = cross_loadresult()
    
    # Plot Flood
    plt_flood_err_trained(res_trained_floodno1, '8-4-1 LR=0.01 MR=0.01')
    plt_flood_err_trained(res_trained_floodno2, '8-4-1 LR=0.05 MR=0.03')
    plt_flood_err_trained(res_trained_floodno3, '8-8-1 LR=0.01 MR=0.01')
    plt_flood_err_trained(res_trained_floodno4, '8-2-2-1 LR=0.01 MR=0.01')
    
    plt_flood_lastepoch(res_trained_floodno1, res_tested_floodno1, '8-4-1 LR=0.01 MR=0.01')
    plt_flood_lastepoch(res_trained_floodno2, res_tested_floodno2, '8-4-1 LR=0.05 MR=0.03')
    plt_flood_lastepoch(res_trained_floodno3, res_tested_floodno3, '8-8-1 LR=0.01 MR=0.01')
    plt_flood_lastepoch(res_trained_floodno4, res_tested_floodno4, '8-2-2-1 LR=0.01 MR=0.01')
    
    er_trained_1 = pd.DataFrame([float(er) for er in res_trained_floodno1[min_idx_floodno1]], index=idx_epoch_flood, columns=['8-4-1    LR=0.01 MR=0.01'])
    er_trained_2 = pd.DataFrame([float(er) for er in res_trained_floodno2[min_idx_floodno2]], index=idx_epoch_flood, columns=['8-4-1    LR=0.05 MR=0.02'])
    er_trained_3 = pd.DataFrame([float(er) for er in res_trained_floodno3[min_idx_floodno3]], index=idx_epoch_flood, columns=['8-8-1    LR=0.01 MR=0.01'])
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
    plt.title('Flood Models RMSE Convergence', fontweight='bold', fontsize='24')
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.show()
    

    # Plot Cross
    plt_cross_cf_matrix(res_trained_crossno1, 'Blues', 'trained', '2-4-2 LR=0.01 MR=0.01')    
    plt_cross_cf_matrix(res_tested_crossno1, 'YlOrBr', 'tested', '2-4-2 LR=0.01 MR=0.01')
    
    plt_cross_cf_matrix(res_trained_crossno2, 'Blues', 'trained', '2-4-2 LR=0.005 MR=0.06')    
    plt_cross_cf_matrix(res_tested_crossno2, 'YlOrBr', 'tested', '2-4-2 LR=0.005 MR=0.06')
    
    plt_cross_cf_matrix(res_trained_crossno3, 'Blues', 'trained', '2-8-2 LR=0.01 MR=0.01')    
    plt_cross_cf_matrix(res_tested_crossno3, 'YlOrBr', 'tested', '2-8-2 LR=0.01 MR=0.01')
    
    plt_cross_cf_matrix(res_trained_crossno4, 'Blues', 'trained','2-4-4-2 LR=0.01 MR=0.01')    
    plt_cross_cf_matrix(res_tested_crossno4, 'YlOrBr', 'tested', '2-4-4-2 LR=0.01 MR=0.01')
         
    ac_trained1 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno1[max_idx_crossno1]], index=idx_epoch_cross, columns=['2-8-2    LR=0.01 MR=0.01'])
    ac_trained1.index.name = 'Epoch'
    
    ac_trained2 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno2[max_idx_crossno2]], index=idx_epoch_cross, columns=['2-8-2    LR=0.005 MR=0.04'])
    ac_trained2.index.name = 'Epoch'
    
    ac_trained3 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno3[max_idx_crossno3]], index=idx_epoch_cross, columns=['2-32-2  LR=0.01 MR=0.01'])
    ac_trained3.index.name = 'Epoch'
    
    ac_trained4 = pd.DataFrame([cm.get_accuracy() for cm in res_trained_crossno4[max_idx_crossno4]], index=idx_epoch_cross, columns=['2-4-4-2 LR=0.01 MR=0.01'])
    ac_trained4.index.name = 'Epoch'
    
    merged_ac = pd.concat([ac_trained1, ac_trained2, ac_trained3, ac_trained4], axis=1)
    ax = sns.lineplot(data=merged_ac)
    ax.lines[0].set_linestyle('solid')
    ax.lines[1].set_linestyle('solid')
    ax.lines[2].set_linestyle('solid')
    ax.lines[3].set_linestyle('solid')
    plt.title('Cross Models Accuracy Convergence', fontweight='bold', fontsize='24')
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.show()