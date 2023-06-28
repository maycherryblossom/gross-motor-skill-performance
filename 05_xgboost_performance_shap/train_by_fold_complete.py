import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier, DMatrix
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import shap
import itertools
import matplotlib.gridspec as gridspec

def main():
    random.seed(0)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    age_idx = 'B'
    method = "my_skeleton"
    save_path = f'05_xgboost_performance_shap/generated/{method}/{age_idx}/'
    shap_each_path = save_path + 'shap/'
    if not os.path.exists(shap_each_path):
        os.makedirs(shap_each_path)

    result_of_new_label = pd.DataFrame(columns=['fold', 'f1_score', 'roc_auc_score', 'precision', 'recall', 'accuracy', 'confusion_matrix'])

    cnt = 0

    fprs = []
    # tprs = []

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    aucs = []

    shap_total = None
    confusions = np.zeros((2, 2))
    

    for i in range (1, 29):
        # if i == 3:
        #     break

        train = pd.read_csv(f'03_gcn_performance/{method}/{age_idx}/fold_{i}/short_train_labeled.csv')
        val = pd.read_csv(f'03_gcn_performance/{method}/{age_idx}/fold_{i}/short_val_labeled.csv')
        test = pd.read_csv(f'03_gcn_performance/{method}/{age_idx}/fold_{i}/short_test_labeled.csv')

        y_train = train["label_pid"]
        X_train = train.drop(columns=["label_pid", 'pid'])

        y_val = val["label_pid"]
        X_val = val.drop(columns=["label_pid", 'pid'])

        y_test = test["label_pid"]
        X_test = test.drop(columns=["label_pid", 'pid'])

        model = XGBClassifier(n_estimators=100, gamma=0.25, learning_rate = 0.3, alpha=0.00015, colsample_bytree=0.75, reg_lambda=1.008)

        model.fit(X_train, y_train, eval_set= [(X_val, y_val)], eval_metric=["error", "auc"], early_stopping_rounds=20)

        predicted = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        fpr = np.interp(base_fpr, fpr, fpr)
        fpr[0] = 0.0
        fprs.append(fpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # tprs.append(tpr)

        explainer = shap.Explainer(model, X_train)
        shap_value = explainer(X_test)
        plt.figure()
        shap.plots.bar(shap_value, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_each_path + f"aggregated_shap_values_fold_{i}.png")

        plt.figure()
        shap.plots.beeswarm(shap_value, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_each_path + f"beeswarm_shap_values_fold_{i}.png")

        if shap_total is None:
            shap_total = shap_value
        else:
            shap_total += shap_value

        xgb_roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        xgb_f1_score = f1_score(y_test, predicted)
        
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)

        confusions += confusion_matrix(y_test, predicted)

        result_of_new_label.loc[cnt] = [i, xgb_f1_score, xgb_roc_score, precision, recall, accuracy, confusion_matrix(y_test, predicted)]
        print(result_of_new_label)

        cnt += 1

    confusions = confusions / 28
    confusions = confusions.astype('float') 
    row_sums = confusions.sum(axis=1) 
    confusions_normalized = confusions / row_sums[:, np.newaxis]
    confusions_transposed = confusions_normalized.T
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(confusions_transposed, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Group A', 'Group B']) 
    plt.yticks(tick_marks, ['Group A', 'Group B'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar(im)  

    for i, j in itertools.product(range(confusions.shape[0]), range(confusions.shape[1])):
        norm = confusions_normalized[i, j]
        plt.text(j, i, f"{confusions[i, j]:.0f} ({norm:.2f})", horizontalalignment="center", color="white" if confusions[i, j] > confusions.max() / 2 else "black")

    plt.savefig(save_path + 'confusion.png', dpi=700)

    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = mean_tpr - std_tpr

    diff = mean_tpr - mean_fpr
    best_index = np.argmax(diff) 
    best_fpr = mean_fpr[best_index]
    best_tpr = mean_tpr[best_index]

    plt.figure()
    plt.title('Aggregated Shap of Folds')
    shap_total /= 28
    shap.plots.beeswarm(shap_total, max_display=20)
    plt.savefig(save_path + 'shap_beeswarm_total_avg.png')

    shap_total_pd = pd.DataFrame(abs(shap_total.values).mean(0), index=X_test.columns)
    shap_total_pd.to_csv(save_path + "/shap_beeswarm_total_avg_data.csv")

    result_of_new_label.to_csv(save_path + 'result_of_new_label.csv')
    
    plt.figure()
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.3)

    axs = []
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])

    axs[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='Standard deviation')
    axs[0].scatter([best_fpr], [best_tpr], marker='o', color="red", label=f"Best cut-off point of Stage 2 model\nSpecificity:{(1-best_fpr):.2f}\nSensitivity:{best_tpr:.2f}")
    axs[0].scatter(1-0.81, 0.91, marker='o', color="yellow", label="Expert panel")
    axs[0].scatter([1-0.29, 1-0.59, 1-0.83], [0.93, 0.94, 0.89], marker='o', color="green", label="Non-expert panel")

    axs[0].plot(mean_fpr, mean_tpr, 'k', label='Stage 2 model(AUC=%0.2f)' % mean_auc, lw=2)
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('1-specificity(%)', fontsize=10)
    axs[0].set_ylabel('Sensitivity(%)', fontsize=10)
    axs[0].legend(loc="lower right", fontsize=10)

    data = pd.read_csv('05_xgboost_performance_shap/generated/my_skeleton/B/shap_beeswarm_total_avg_data.csv')
    columns = np.array(data['Unnamed: 0'])
    new_data = pd.DataFrame(np.array(data['0']).reshape(1, 12), columns=columns)
    new_data['act_1'] = (new_data['1_1'] + new_data['1_2'] + new_data['1_3']) 
    new_data['act_2'] = (new_data['2_1'] + new_data['2_2'] + new_data['2_3']) 
    new_data['act_3'] = (new_data['3_1'] + new_data['3_2'] + new_data['3_3']) 
    new_data['act_4'] = (new_data['4_1'] + new_data['4_2'] + new_data['4_3']) 

    final = new_data[['act_1', 'act_2', 'act_3', 'act_4']]
    values = final.loc[0].tolist()
    sns.barplot(x=values, y=['Climb up the stairs', 'Go down the stairs', 'Throw the ball', 'Stand on one foot'], orient='h', color=sns.xkcd_rgb['neon red'])
    axs[-1].set_xlabel('Shap value')
    for i in range(len(values)):
        axs[-1].text(values[i]+0.01, i, f"+ {values[i]:.2f}", va='center', ha='left', color=sns.xkcd_rgb['neon red'], fontsize=10)
    axs[-1].set_xlim(0, max(values)*1.2)
    plt.savefig(save_path + 'Figure 4.pdf', dpi=300, transparent=True, format='pdf',bbox_inches='tight', pad_inches=0)
   
main()