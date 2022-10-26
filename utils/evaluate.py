from models.utils import custom_replace
from utils.metrics import *
import torch.nn.functional as F
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(args,
                    all_predictions,
                    all_targets,
                    elapsed=0,
                    verbose=True):

    meanAP = skmetrics.average_precision_score(all_targets, all_predictions, average='macro', pos_label=1)

    optimal_threshold = 0.5 

    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()
    
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = skmetrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = skmetrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    OP = skmetrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = skmetrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)  

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))        
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', OF1),
                        ('Label-based Macro F1', CF1)])

    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('mAP:   {:0.1f}'.format(meanAP*100))
        print('----')
        print('CP:    {:0.1f}'.format(CP*100))
        print('CR:    {:0.1f}'.format(CR*100))
        print('CF1:   {:0.1f}'.format(CF1*100))
        print('OP:    {:0.1f}'.format(OP*100))
        print('OR:    {:0.1f}'.format(OR*100))
        print('OF1:   {:0.1f}'.format(OF1*100))

    metrics_dict = {}
    metrics_dict['mAP'] = meanAP
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1
    metrics_dict['time'] = elapsed

    print('')

    return metrics_dict