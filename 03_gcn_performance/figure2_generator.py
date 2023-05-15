import numpy as np
import pandas as pd
import pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, plot_roc_curve, recall_score, precision_score
import itertools
import matplotlib.gridspec as gridspec

def main():
    # plt.rcParams['font.family'] = 'Times New Roman'plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    
    result = pd.DataFrame(columns=['fold', 'action', 'weighted_f1_score', 'micro_f1_score', 'macro_f1_score', 'roc_auc_score', 'accuracy', 'recall', 'precision', 'sensitivity', 'specificity','confusion_matrix'])
    cnt = 0

    age_idx='B'
    
    method = "my_skeleton"

    roc_curves = []
    tprs = []
    aucs = []

    all_fpr = np.linspace(0, 1, 101)
    mean_tprs_action_1 = []
    mean_tprs_action_2 = []
    mean_tprs_action_3 = []
    mean_tprs_action_4 = []

    fprs_action_1 = []
    fprs_action_2 = []
    fprs_action_3 = []
    fprs_action_4 = []

    aucs_action_1 = []
    aucs_action_2 = []
    aucs_action_3 = []
    aucs_action_4 = []

    aucs2 = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    accuracies = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    recalls = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    precisions = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    sensitivities = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    specificities = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    f1s = {'action_1': 0.0, 'action_2': 0.0, 'action_3': 0.0, 'action_4': 0.0}
    confusions = {'action_1': np.zeros((3, 3)), 'action_2': np.zeros((3, 3)), 'action_3': np.zeros((3, 3)), 'action_4': np.zeros((3, 3))}


    for i in range (1, 29):
        print(i)
        # if i == 1:
        #     continue

        for j in range(4):

            with open(f"work_dirs/{method}/{age_idx}/{j+1}/fold_{i}/best_pred.pkl","rb") as fr:
                data = pickle.load(fr)

            with open(f"01_skeleton_extraction/{method}/{age_idx}/{j+1}/fold_{i}_test.pkl","rb") as fr:
                answer = pickle.load(fr)

            trueLi = []

            for k in answer:
                trueLi.append(int(k['label']))

            tarr = np.array(trueLi)

            preds = np.argmax(data, axis=1)
            conf = confusion_matrix(trueLi, preds) 

            roc_score = roc_auc_score(trueLi, data, multi_class='ovr', average='macro')
            weighted_f1 = f1_score(tarr, preds, average="weighted")
            micro_f1 = f1_score(tarr, preds, average="micro")
            macro_f1 = f1_score(tarr, preds, average="macro")
            accuracy = accuracy_score(tarr, preds)
            recall = recall_score(trueLi, preds, average='macro')
            precision = precision_score(trueLi, preds, average='macro')
            sensitivity = recall_score(trueLi, preds, average='macro', labels=[0])
            specificity = recall_score(trueLi, preds, average='macro', labels=[1,2])

            fpr, tpr, roc_auc = dict(), dict(), dict()
            n_classes = 3
            
            data_np = np.array(data)
            for k in range(n_classes):
                fpr[k], tpr[k], _ = roc_curve(tarr == k, data_np[:, k])
                roc_auc[k] = auc(fpr[k], tpr[k])

            fpr_grid = np.linspace(0.0, 1.0, 100)
            # fpr, tpr, _ = roc_curve(tarr, preds)
            mean_tpr = np.zeros_like(fpr_grid)
            
            for k in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[k], tpr[k])

            mean_tpr /= n_classes
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            print(auc(fpr_grid, mean_tpr)) # 이거 roc_auc_score(trueLi, data, multi_class='ovr', average='macro') 이거랑 같다~

            if (j+1 == 1):
                fprs_action_1.append(fpr_grid)
                mean_tprs_action_1.append(mean_tpr)
                aucs_action_1.append(auc(fpr_grid, mean_tpr))
                aucs2['action_1'] += auc(fpr_grid, mean_tpr)
                accuracies['action_1'] += accuracy
                recalls['action_1'] += recall
                precisions['action_1'] += precision
                sensitivities['action_1'] += sensitivity
                specificities['action_1'] += specificity
                f1s['action_1'] += macro_f1
                confusions['action_1'] += conf

            elif (j+1 == 2):
                fprs_action_2.append(fpr_grid)
                mean_tprs_action_2.append(mean_tpr)
                aucs_action_2.append(auc(fpr_grid, mean_tpr))
                aucs2['action_2'] += auc(fpr_grid, mean_tpr)
                accuracies['action_2'] += accuracy
                recalls['action_2'] += recall
                precisions['action_2'] += precision
                sensitivities['action_2'] += sensitivity
                specificities['action_2'] += specificity
                f1s['action_2'] += macro_f1
                confusions['action_2'] += conf

            elif (j+1 == 3):
                fprs_action_3.append(fpr_grid)
                mean_tprs_action_3.append(mean_tpr)
                aucs_action_3.append(auc(fpr_grid, mean_tpr))
                aucs2['action_3'] += auc(fpr_grid, mean_tpr)
                accuracies['action_3'] += accuracy
                recalls['action_3'] += recall
                precisions['action_3'] += precision
                sensitivities['action_3'] += sensitivity
                specificities['action_3'] += specificity
                f1s['action_3'] += macro_f1
                confusions['action_3'] += conf

            elif (j+1 == 4):
                fprs_action_4.append(fpr_grid)
                mean_tprs_action_4.append(mean_tpr)
                aucs_action_4.append(auc(fpr_grid, mean_tpr))
                aucs2['action_4'] += auc(fpr_grid, mean_tpr)
                accuracies['action_4'] += accuracy
                recalls['action_4'] += recall
                precisions['action_4'] += precision
                sensitivities['action_4'] += sensitivity
                specificities['action_4'] += specificity
                f1s['action_4'] += macro_f1
                confusions['action_4'] += conf

            result.loc[cnt] = [i, j+1, weighted_f1, micro_f1, macro_f1, roc_score, accuracy, recall, precision, sensitivity, specificity, conf]
            cnt += 1

    confusions['action_1'] = confusions['action_1'].astype('float') / confusions['action_1'].sum(axis=1)[:, np.newaxis]
    confusions['action_2'] = confusions['action_2'].astype('float') / confusions['action_2'].sum(axis=1)[:, np.newaxis]
    confusions['action_3'] = confusions['action_3'].astype('float') / confusions['action_3'].sum(axis=1)[:, np.newaxis]
    confusions['action_4'] = confusions['action_4'].astype('float') / confusions['action_4'].sum(axis=1)[:, np.newaxis]

    metrics = ['AUC', 'Accuracy', 'F1-score']
    actions = ['action_1', 'action_2', 'action_3', 'action_4']

    data = {
        'Climb up the stairs': (aucs2['action_1']/28, accuracies['action_1']/28, f1s['action_1']/28),
        'Go down the stairs': (aucs2['action_2']/28, accuracies['action_2']/28, f1s['action_2']/28),
        'Throw the ball': (aucs2['action_3']/28, accuracies['action_3']/28, f1s['action_3']/28),
        'Stand on one foot': (aucs2['action_4']/28, accuracies['action_4']/28, f1s['action_4']/28)
    }

    action_names = list(data.keys())
    values = np.array(list(data.values()))
    stds = values.std(axis=1)

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

    axs = []
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)
    axs[-1].spines['left'].set_visible(False)
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])

    width = 0.2
    x = np.arange(len(metrics))

    colors = ['lightsalmon', 'mediumaquamarine', 'cornflowerblue', 'plum']

    for i in range(len(action_names)):
        axs[0].bar(x + (i * width), values[i], width, label=action_names[i], yerr=stds[i], color=colors[i], capsize=10)
        for j in range(len(metrics)):
            axs[0].text(x[j] + (i * width), values[i, j] + 0.02, f'{values[i, j]:.2f}', ha='center', fontsize=10)
    axs[0].set_xticks(x + ((len(action_names) - 1) * width) / 2)
    axs[0].set_xticklabels(metrics, fontsize=10)
    axs[0].legend()

    tick_marks = np.arange(3)
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[1], wspace=0.7, hspace=0.3)

    for i in range(len(action_names)):
        axs.append(fig.add_subplot(gs1[i // 2, i % 2])) 
        axs[-1].imshow(confusions[actions[i]], interpolation='nearest', cmap=plt.cm.Blues)
        for j, k in itertools.product(range(confusions[actions[i]].shape[0]), range(confusions[actions[i]].shape[1])):
            axs[-1].text(k, j, format(confusions[actions[i]][j, k], '.2f'), horizontalalignment='center', color='white' if confusions[actions[i]][j, k] > confusions[actions[i]].max() / 2 else 'black', fontsize=10)
        axs[-1].set_yticks(tick_marks)
        axs[-1].set_yticklabels(['Bad', 'Good', 'Perfect'], fontsize=10)
        axs[-1].set_xticks(tick_marks)
        axs[-1].set_xticklabels(['Bad', 'Good', 'Perfect'], fontsize=10)
        axs[-1].set_ylabel('Actual')
        axs[-1].set_xlabel('Predicted')
        axs[-1].set_title(f'{action_names[i]}', fontsize=10)

    plt.savefig(f'04_gcn_performance/generated/{method}/{age_idx}/scores.pdf',bbox_inches='tight', pad_inches=0, dpi=300, transparent=True, format='pdf')

    fpr_folds_mean_action_1 = np.mean(fprs_action_1, axis=0)
    tpr_folds_mean_action_1 = np.mean(mean_tprs_action_1, axis=0)
    auc_folds_mean_action_1 = np.mean(aucs_action_1)

    fpr_folds_mean_action_2 = np.mean(fprs_action_2, axis=0)
    tpr_folds_mean_action_2 = np.mean(mean_tprs_action_2, axis=0)
    auc_folds_mean_action_2 = np.mean(aucs_action_2)

    fpr_folds_mean_action_3 = np.mean(fprs_action_3, axis=0)
    tpr_folds_mean_action_3 = np.mean(mean_tprs_action_3, axis=0)
    auc_folds_mean_action_3 = np.mean(aucs_action_3)

    fpr_folds_mean_action_4 = np.mean(fprs_action_4, axis=0)
    tpr_folds_mean_action_4 = np.mean(mean_tprs_action_4, axis=0)
    auc_folds_mean_action_4 = np.mean(aucs_action_4)

    std_tpr_action_1 = np.std(mean_tprs_action_1, axis=0)
    tprs_upper_action_1 = np.minimum(tpr_folds_mean_action_1 + std_tpr_action_1, 1)
    tprs_lower_action_1 = np.maximum(tpr_folds_mean_action_1 - std_tpr_action_1, 0)

    std_tpr_action_2 = np.std(mean_tprs_action_2, axis=0)
    tprs_upper_action_2 = np.minimum(tpr_folds_mean_action_2 + std_tpr_action_2, 1)
    tprs_lower_action_2 = np.maximum(tpr_folds_mean_action_2 - std_tpr_action_2, 0)

    std_tpr_action_3 = np.std(mean_tprs_action_3, axis=0)
    tprs_upper_action_3 = np.minimum(tpr_folds_mean_action_3 + std_tpr_action_3, 1)
    tprs_lower_action_3 = np.maximum(tpr_folds_mean_action_3 - std_tpr_action_3, 0)

    std_tpr_action_3 = np.std(mean_tprs_action_3, axis=0)
    tprs_upper_action_3 = np.minimum(tpr_folds_mean_action_3 + std_tpr_action_3, 1)
    tprs_lower_action_3 = np.maximum(tpr_folds_mean_action_3 - std_tpr_action_3, 0)

    std_tpr_action_4 = np.std(mean_tprs_action_4, axis=0)
    tprs_upper_action_4 = np.minimum(tpr_folds_mean_action_4 + std_tpr_action_4, 1)
    tprs_lower_action_4 = np.maximum(tpr_folds_mean_action_4 - std_tpr_action_4, 0)

    plt.figure()
    plt.plot(fpr_folds_mean_action_1, tpr_folds_mean_action_1, color='hotpink', label='Climb up the stairs (AUC = %0.2f)' % auc_folds_mean_action_1, lw=2)
    plt.fill_between(fpr_folds_mean_action_1, tprs_lower_action_1, tprs_upper_action_1, color='lightpink', alpha=.35,
                    label=r'$\pm$ 1 std. dev.')
    plt.plot(fpr_folds_mean_action_2, tpr_folds_mean_action_2, color='darkorange', label='Go down the stairs (AUC = %0.2f)' % auc_folds_mean_action_2, lw=2)
    plt.fill_between(fpr_folds_mean_action_2, tprs_lower_action_2, tprs_upper_action_2, color='peachpuff', alpha=.35,
                    label=r'$\pm$ 1 std. dev.')
    plt.plot(fpr_folds_mean_action_3, tpr_folds_mean_action_3, color='gold', label='Throw the ball (AUC = %0.2f)' % auc_folds_mean_action_3, lw=2)
    plt.fill_between(fpr_folds_mean_action_3, tprs_lower_action_3, tprs_upper_action_3, color='yellow', alpha=.35,
                    label=r'$\pm$ 1 std. dev.')
    plt.plot(fpr_folds_mean_action_4, tpr_folds_mean_action_4, color='seagreen', label='Stand on one foot (AUC = %0.2f)' % auc_folds_mean_action_4, lw=2)
    plt.fill_between(fpr_folds_mean_action_4, tprs_lower_action_4, tprs_upper_action_4, color='mediumseagreen', alpha=.35,
                    label=r'$\pm$ 1 std. dev.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'04_gcn_performance/generated/{method}/{age_idx}/roc_curves_by_action.png', dpi=700)
    result.to_csv(f"04_gcn_performance/generated/{method}/{age_idx}/each_action_eval.csv")

main()  