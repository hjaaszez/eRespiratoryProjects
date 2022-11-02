import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

class Evaluation_icbhi():
    '''
    parameter
    predict_label : モデルの出力結果
    true_label : 正解ラベル 
    result_dir : 結果格納フォルダ
    result_name : 実験の名前
    SE:Sensitivity
    SP:Specifisity
    AS:Average Score
    HS:Hermonic Score
    '''
    def __init__(self, predict_label, true_label, result_dir, result_name) :
        self.predict_label = predict_label 
        self.true_label    = true_label
        self.result_dir    = result_dir
        self.result_name   = result_name
        self.result        = [0]*4
        self.auc_result    = [0]*4
    
    '''
    calculate ICBHI Score
    '''
    def calculate_evaluation(self):
        pred = np.identity(4)[np.argmax(self.predict_label, axis = 1)]
        
        num_normal_predict   = np.count_nonzero(self.true_label[:, 0] + pred[:, 0] == 2)
        num_normal           = np.count_nonzero(self.true_label[:, 0] == 1)
        num_abnormal_predict = np.count_nonzero(self.true_label[:, 1] + pred[:, 1] == 2) + \
                               np.count_nonzero(self.true_label[:, 2] + pred[:, 2] == 2) + \
                               np.count_nonzero(self.true_label[:, 3] + pred[:, 3] == 2)
        num_abnormal         = np.count_nonzero(self.true_label[:, 1] == 1) + \
                               np.count_nonzero(self.true_label[:, 2] == 1) + \
                               np.count_nonzero(self.true_label[:, 3] == 1)

        SE = num_abnormal_predict / num_abnormal
        SP = num_normal_predict   / num_normal
        AS = (SE + SP) / 2
        HS = 2 * SE * SP / (SE + SP)
        # self.result[0] += SE
        # self.result[1] += SP
        # self.result[2] += AS
        # self.result[3] += HS
                
        print("SE = ", SE)
        print("SP = ", SP)
        print("AS = ", AS)
        print("HS = ", HS)
    
    '''
    calculate ROC-AUC(four class)
    '''
    def auc_four_class(self):
        folder = ["normal", "crackle", "wheeze", "crackle_wheeze"]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(folder)):
            fpr[i], tpr[i], _ = roc_curve(self.true_label[:, i], self.predict_label[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(len(folder)):
            #plt.plot(fpr[i], tpr[i], label='ROC curve of class %s (area = %.2f)' % (folder[i], roc_auc[i]))
            #print("AUC of class %s = %f" % (folder[i], roc_auc[i]))
            self.auc_result[i] += roc_auc[i]
        
        # plt.plot([0, 1], [0, 1])
        # plt.xlim([-0.05, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend()
        # plt.grid(True)
        # savedir = f'{self.result_dir}/{self.result_name}'
        # os.makedirs(savedir, exist_ok=True)
        # plt.savefig(f"{savedir}/roc.png")
        # plt.close()
        # plt.show()
    '''
    calculate ROC-AUC(two class: normal,abnormal)
    '''
    def auc_two_class(self, plot=False):
        predict    = np.argmax(self.predict_label, axis=1)
        true_label = np.argmax(self.true_label, axis=1)

        predict_2class    = np.zeros(len(true_label))
        true_label_2class = np.zeros(len(true_label), dtype=int)

        for i in range(len(true_label)):
            if true_label[i] != 0:
                true_label_2class[i] = 1
            predict_2class[i] = self.predict_label[i,1] + self.predict_label[i,2] + self.predict_label[i,3]
        
        fpr, tpr, _ = roc_curve(true_label_2class, predict_2class)
        AUC = auc(fpr, tpr)
        print('--------------------AUC--------------------')
        print(f'AUC={AUC}')
        
        print('------------------save ROC-----------------')
        plt.plot(fpr, tpr, label='%s (AUC: %.2f)' % (os.path.basename(self.result_dir), AUC))       
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.grid(True)
        savedir = f'{self.result_dir}/result_roc'
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(f"{savedir}/roc.png")
        print(f'saved ({savedir}/roc.png)')
        #plt.close()
        #plt.show()



