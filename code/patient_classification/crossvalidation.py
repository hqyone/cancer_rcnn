# This file is used for cell classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix,average_precision_score,precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
import xgboost as xgb
from xgboost import plot_importance, to_graphviz
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Read the data for training and validation
patient_data_dir = "/home/hqyone/mnt/2tb/github/cancer_rcnn/data/patent_predict"

mod1_yang_file = f"{patient_data_dir}/yang-211-model-T1.tsv"
mod1_yin_file = f"{patient_data_dir}/yin-189-model-T1.tsv"
mod4_yang_file = f"{patient_data_dir}/yang-211-model-A3.tsv"
mod4_yin_file = f"{patient_data_dir}/yin-189-model-A3.tsv"

mod1_yang_df = pd.read_csv(mod1_yang_file, sep='\t')
mod1_yang_df['type']=1
mod1_yin_df = pd.read_csv(mod1_yin_file, sep='\t')
mod1_yin_df['type']=0
mod4_yang_df = pd.read_csv(mod4_yang_file, sep='\t')
mod4_yang_df['type']=1
mod4_yin_df = pd.read_csv(mod4_yin_file, sep='\t')
mod4_yin_df['type']=0

mod1_df = mod1_yang_df.append(mod1_yin_df)
mod4_df = mod4_yang_df.append(mod4_yin_df)

def getXY(df):
    x = df.drop(columns=['patientID','type'], axis=1)
    #x = df.iloc[:,:-1]
    x['yang_ratio']= x['yang']/(x['yang']+x['yin-yang']+x['yin'])
    x['comb_yang']= x['yang']+x['yin-yang']
    x['yang_ratio2']= (x['yang']+x['yin-yang'])/(x['yang']+x['yin-yang']+x['yin'])  
    y = df['type']
    return(x,y)
  
def plot_pr(recall,precision,average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: {0:0.6f}'.format(average_precision))
    plt.show()
    
def plot_roc(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (area = %0.6f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.show()

def getQCfeatures(y_ground, y_predict):
    cmatrix = confusion_matrix(y_ground, y_predict).ravel()
    return(getQCfeatures_1(cmatrix))

# calculate specificity, sensitivity, precision, recall
# ppv, npv and F1 score based on a confusion matrix
def getQCfeatures_1(cmatrix):
    tn =cmatrix[0]
    fp =cmatrix[1]
    fn =cmatrix[2]
    tp =cmatrix[3]
    specificity = tn/(tn+fp)*100
    sensitivity = tp/(tp+fn)*100
    precision = tp/(tp+fp) * 100
    recall = tp/(tp+fn) * 100
    ppv = tp/(tp+fp)*100
    npv = tn/(fn+tn)*100
    F1= 2*(precision*recall)/(precision+recall)
    return(specificity, sensitivity, precision, recall, ppv, npv, F1)

# x is predicted features and y is outcome
# model_type is a string from [logistic, svm, randomforest, xgb]
def CrossValidation(x, y, model_type, n_splits=10, seed=72):
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #skf.get_n_splits(x, y)
    
    total_test_accuracy = []
    total_test_sensitivity = []
    total_test_specificity = []
    total_test_precision = []
    total_test_recall = []
    total_test_ppv = []
    total_test_npv = []
    total_test_F1 = []
    total_test_auc_roc = []
    total_average_precision = []
    combined_confusion_matrix = [[0,0],[0,0]]
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Select models
        model1 = None
        # possible methods : logistic, svm, randomforest, xgb
        if (model_type=="logistic"):
            scaler = StandardScaler()
            lr = LogisticRegression()
            model1 = Pipeline([('standardize', scaler),
                                ('log_reg', lr)])
            model1.fit(X_train, y_train)
        elif (model_type == "svm"):
            model1 = svm.LinearSVC()
            model1.fit(X_train, y_train)
        elif (model_type == "randomforest"):
            model1 = RandomForestClassifier(n_estimators=10, random_state=42)
            model1.fit(X_train, y_train)
        elif (model_type == "xgb"):
            learning_rate =0.05
            w=1
            model1 = xgb.XGBClassifier(
                        learning_rate=learning_rate,
                        n_estimators=1000,
                        max_depth=4,
                        min_child_weight=4,
                        gamma=0.6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=5e-05,
                        objective='binary:logistic',
                        nthread=20,
                        scale_pos_weight=w,
                        seed=27)
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model1.fit(X_train,
                    y_train,
                    early_stopping_rounds=25,
                    eval_metric=['auc','error','logloss'],
                    eval_set=eval_set,
                    verbose=False)
        else:
            print(f"The model_type {model_type} is not expected, exit!")
        if model1:
            y_train_hat = model1.predict(X_train)
            y_train_hat_probs = None
            if model_type=="svm":
                y_train_hat_probs = model1._predict_proba_lr(X_train)[:,1]
                y_test_hat_probs = model1._predict_proba_lr(X_test)[:,1]
                y_pred_proba = model1._predict_proba_lr(X_test)[::,1]
            else:
                y_train_hat_probs = model1.predict_proba(X_train)[:,1]
                y_test_hat_probs = model1.predict_proba(X_test)[:,1]
                y_pred_proba = model1.predict_proba(X_test)[::,1]
                

            train_accuracy = accuracy_score(y_train, y_train_hat)*100
            train_auc_roc = roc_auc_score(y_train, y_train_hat_probs)*100
            

            y_test_hat = model1.predict(X_test)

            test_accuracy = accuracy_score(y_test, y_test_hat)*100
            test_specificity, test_sensitivity, test_precision, test_recall, test_ppv, test_npv, test_F1 = getQCfeatures(y_test, y_test_hat)
            
            test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100
            total_test_auc_roc.append(test_auc_roc)
            
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

            y_scores = y_pred_proba
            roc_auc = auc(fpr, tpr)
            
            average_precision = average_precision_score(y_test, y_scores)
            total_average_precision.append(average_precision)
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            
            total_test_accuracy.append(test_accuracy)
            total_test_sensitivity.append(test_sensitivity)
            total_test_specificity.append(test_specificity)
            total_test_precision.append(test_precision)
            total_test_recall.append(test_recall)
            total_test_ppv.append(test_ppv)
            total_test_npv.append(test_npv)
            total_test_F1.append(test_F1)
            combined_confusion_matrix = np.add(combined_confusion_matrix, confusion_matrix(y_train, y_train_hat))

    plot_roc(fpr, tpr, roc_auc)
    plot_pr(recall,precision,average_precision)

    #create ROC curve
    # plt.plot(fpr,tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    report_str =""
    if len(total_test_accuracy)>0:
        print(f'{model_type}\n')
        print('Confusion matrix:\n', confusion_matrix(y_train, y_train_hat))
        print('Combined_Confusion matrix:\n', combined_confusion_matrix)
        print('specificity, sensitivity, precision, recall, ppv, npv, F1:', getQCfeatures_1(combined_confusion_matrix.ravel()))
        print('Training AUC: %.4f %%' % train_auc_roc)
        print('Training accuracy: %.4f %%' % train_accuracy)

        print('Test Confusion matrix:\n', confusion_matrix(y_test, y_test_hat))
        print('Test Training AUC: %.4f %%' % test_auc_roc)
        print('Test Training accuracy: %.4f %%' % test_accuracy)
        print('Average_precision: %.4f ' % float(average_precision))

        print(f'total_test_accuracy:{np.mean(total_test_accuracy)},{np.std(total_test_accuracy)}')
        print(f'total_test_specificity:{np.mean(total_test_specificity)},{np.std(total_test_specificity)}')
        print(f'total_test_sensitivity:{np.mean(total_test_sensitivity)},{np.std(total_test_sensitivity)}')
        print(f'total_test_precision:{np.mean(total_test_precision)},{np.std(total_test_precision)}')
        print(f'total_test_recall:{np.mean(total_test_recall)},{np.std(total_test_recall)}')
        print(f'total_test_ppv:{np.mean(total_test_ppv)},{np.std(total_test_ppv)}')
        print(f'total_test_npv:{np.mean(total_test_npv)},{np.std(total_test_npv)}')
        print(f'total_test_F1:{np.mean(total_test_F1)},{np.std(total_test_F1)}')
        
        report_str+=f'{round(train_auc_roc,3)}\t{round(train_accuracy,3)}\t'
        report_str+=f'{round(np.mean(total_test_auc_roc),3)},{round(np.std(total_test_auc_roc),3)}\t'
        report_str+=f'{round(np.mean(total_average_precision),3)},{round(np.std(total_average_precision),3)}\t'
        report_str+=f'{round(np.mean(total_test_accuracy),3)},{round(np.std(total_test_accuracy),3)}\t'
        report_str+=f'{round(np.mean(total_test_specificity),3)},{round(np.std(total_test_specificity),3)}\t'
        report_str+=f'{round(np.mean(total_test_sensitivity),3)},{round(np.std(total_test_sensitivity),3)}\t'
        report_str+=f'{round(np.mean(total_test_precision),3)},{round(np.std(total_test_precision),3)}\t'
        report_str+=f'{round(np.mean(total_test_recall),3)},{round(np.std(total_test_recall),3)}\t'
        report_str+=f'{round(np.mean(total_test_ppv),3)},{round(np.std(total_test_ppv),3)}\t'
        report_str+=f'{round(np.mean(total_test_npv),3)},{round(np.std(total_test_npv),3)}\t'
        report_str+=f'{round(np.mean(total_test_F1),3)},{round(np.std(total_test_F1),3)}'

        data_directory = {
            "auc":total_test_auc_roc,
            "ppv":total_test_ppv,
            "npv":total_test_npv,
            "accuracy":total_test_accuracy,
            "specificity":total_test_specificity,
            "sensitivity":total_test_sensitivity,
            "F1":total_test_F1
        }

    return (report_str, data_directory)

def runReg_CrossValidation(df,model_type,n_splits=10, seed=120):
    (x, y) =getXY(df)
    return(CrossValidation(x, y,model_type,n_splits, seed))

from scipy import stats
# possible methods : logistic, svm, randomforest, xgb
runReg_CrossValidation(mod1_df, "svm",seed=200)


def get_performance_feature_dic(cell_classify_df, method_ls, out_file):
    with open(out_file,'w') as OUT:
        OUT.write("method\ttrain_auc_roc\ttrain_accuracy\ttest_auc_roc\taverage_precision\ttotal_test_accuracy\ttotal_test_specificity\ttotal_test_sensitivity\ttotal_test_precision\ttotal_test_recall\ttotal_test_ppv\ttotal_test_npv\ttotal_test_F1\n")
        performance_feature_dic={}
        for method in method_ls:
            [report_str,data_directory] = runReg_CrossValidation(cell_classify_df, method,n_splits=10, seed=200)
            performance_feature_dic[method] = data_directory
            OUT.write(method+"\t"+report_str+'\n')
    return performance_feature_dic


def create_patient_predicton_testing_matrix(test_type, performance_feature_dic):
    feature_ls = ["accuracy","specificity","sensitivity",'auc',"F1"]
    comparison_pairs = [("xgb","svm"),("xgb","logistic"),("randomforest","svm"),("randomforest","logistic"),("xgb","randomforest"),("svm","logistic")]
    result_dataframe =pd.DataFrame()
    dic = {}
    for comparison in comparison_pairs:
        p_ls = []
        for feature in feature_ls:
            values_1=performance_feature_dic[comparison[0]][feature]
            values_2=performance_feature_dic[comparison[1]][feature]
            if sum(values_1)-sum(values_2)==0:
                p_val =1
            else:
                if test_type == "wilcoxon":
                    (state, p_val) = stats.wilcoxon(values_1, values_2, alternative='greater')
                elif test_type =="t":
                    (state, p_val) = stats.ttest_ind(values_1, values_2, alternative='greater')
                elif test_type == "mannwhitneyu":
                    (state, p_val) = stats.mannwhitneyu(values_1, values_2, alternative='greater')
            p_ls.append(p_val)
        dic["_".join(comparison)] = p_ls
    result_df = pd.DataFrame.from_dict(dic)
    result_df.index = feature_ls
    return(result_df)

def create_friedman_testing_matrix(test_type, performance_feature_dic):
    feature_ls = ["accuracy","specificity","sensitivity",'auc',"F1"]
    models = ["xgb","logistic","svm","randomforest"]
    p_dic = {}
    for feature in feature_ls:
        (state, p_val) = stats.friedmanchisquare(performance_feature_dic[models[0]][feature], performance_feature_dic[models[1]][feature],
                                                performance_feature_dic[models[2]][feature], performance_feature_dic[models[3]][feature])
        p_dic[feature]=[p_val]
    result_df = pd.DataFrame.from_dict(p_dic)
    return(result_df)

out_file="/home/hqyone/mnt/2tb/github/cancer_rcnn/data/output/patient_predict/summary_mod4.tsv"
method_ls = ["logistic", "svm", "randomforest", "xgb"]
# cell_classify_df = mod4_df
cell_classify_df = mod1_df
performance_feature_dic = get_performance_feature_dic(cell_classify_df, method_ls,out_file)
p_df = create_patient_predicton_testing_matrix("wilcoxon",performance_feature_dic)
print("--------------------wilcoxon-------------------")
print(p_df)

p_df = create_patient_predicton_testing_matrix("mannwhitneyu",performance_feature_dic)
print("--------------------mannwhitneyu-------------------")
print(p_df)


p_df = create_friedman_testing_matrix("friedmanchisquare",performance_feature_dic)
print("--------------------friedmanchisquare-------------------")
print(p_df)




# # Mann-Whitney U test
# print(mod4_data_dic["xgb"]["sensitivity"])
# print(mod4_data_dic["svm"]["sensitivity"])
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["accuracy"], mod4_data_dic["svm"]["accuracy"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# #print(stats.mannwhitneyu(mod4_data_dic["xgb"]["ppv"], mod4_data_dic["svm"]["ppv"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["npv"], mod4_data_dic["svm"]["npv"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"],alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"],alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"],alternative='greater'))
# # wilcoxon
# print("########################### xgb #############################")
# print(stats.wilcoxon(mod4_data_dic["xgb"]["accuracy"], mod4_data_dic["svm"]["accuracy"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["auc"], mod4_data_dic["svm"]["auc"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"], alternative='greater'))

# print(stats.wilcoxon(mod4_data_dic["xgb"]["accuracy"], mod4_data_dic["logistic"]["accuracy"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["logistic"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["sensitivity"], mod4_data_dic["logistic"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["auc"], mod4_data_dic["logistic"]["auc"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))

# print("########################### randomforest #############################")
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["accuracy"], mod4_data_dic["svm"]["accuracy"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["auc"], mod4_data_dic["svm"]["auc"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["svm"]["F1"], alternative='greater'))

# print(stats.wilcoxon(mod4_data_dic["randomforest"]["accuracy"], mod4_data_dic["logistic"]["accuracy"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["specificity"], mod4_data_dic["logistic"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["sensitivity"], mod4_data_dic["logistic"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["auc"], mod4_data_dic["logistic"]["auc"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))

# print(stats.wilcoxon(mod4_data_dic["randomforest"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["auc"], mod4_data_dic["svm"]["auc"], alternative='greater'))


# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["svm"]["F1"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))

# # t-test
# print(stats.ttest_ind(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["auc"], mod4_data_dic["svm"]["auc"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"]))




# out_file="/home/hqyone/mnt/2tb/github/cancer_rcnn/data/output/patient_predict/summary_mod1.tsv"
# with open(out_file,'w') as OUT:
#     methods = ["logistic", "svm", "randomforest", "xgb"]
#     OUT.write("method\ttrain_auc_roc\ttrain_accuracy\ttest_auc_roc\taverage_precision\ttotal_test_accuracy\ttotal_test_specificity\ttotal_test_sensitivity\ttotal_test_precision\ttotal_test_recall\ttotal_test_ppv\ttotal_test_npv\ttotal_test_F1\n")
    
#     mod4_data_dic={}
#     for method in methods:
#         [report_str,data_directory] = runReg_CrossValidation(mod1_df, method,n_splits=10, seed=200)
#         mod4_data_dic[method] = data_directory
#         OUT.write(method+"\t"+report_str+'\n')

# print(mod4_data_dic["xgb"]["sensitivity"])
# print(mod4_data_dic["svm"]["sensitivity"])
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# #print(stats.mannwhitneyu(mod4_data_dic["xgb"]["ppv"], mod4_data_dic["svm"]["ppv"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["npv"], mod4_data_dic["svm"]["npv"], alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"],alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"],alternative='greater'))
# print(stats.mannwhitneyu(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"],alternative='greater'))

# # wilcoxon
# print(stats.wilcoxon(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["sensitivity"], mod4_data_dic["svm"]["sensitivity"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["auc"], mod4_data_dic["svm"]["auc"], alternative='greater'))

# print(stats.wilcoxon(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["svm"]["F1"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))
# print(stats.wilcoxon(mod4_data_dic["randomforest"]["F1"], mod4_data_dic["logistic"]["F1"], alternative='greater'))

# # t-test
# print(stats.ttest_ind(mod4_data_dic["xgb"]["specificity"], mod4_data_dic["svm"]["specificity"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["auc"], mod4_data_dic["svm"]["auc"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["F1"], mod4_data_dic["svm"]["F1"]))
# print(stats.ttest_ind(mod4_data_dic["xgb"]["F1"], mod4_data_dic["logistic"]["F1"]))