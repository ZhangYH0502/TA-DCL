from utils.metrics import *
import warnings
warnings.filterwarnings("ignore")


class LossLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        open(model_name+'/train.log', "w").close()
        open(model_name+'/valid.log', "w").close()
        open(model_name+'/test.log', "w").close()

    def log_losses(self, file_name, epoch, loss, metrics):
        log_file = open(self.model_name+'/'+file_name, "a")
        log_file.write(str(epoch)+','+str(metrics['mAP'])+'\n')
        log_file.close()


class Logger:
    def __init__(self, args):
        self.model_name = args.model_name
        self.best_mAP = 0
        self.best_class_acc = 0

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            # self.file_names = {'train': os.path.join(args.model_name, 'train_results.csv'),
            #                    'valid': os.path.join(args.model_name, 'valid_results.csv'),
            #                    'test': os.path.join(args.model_name, 'test_results.csv'),
            #                    'valid_all_aupr': os.path.join(args.model_name, 'valid_all_aupr.csv'),
            #                    'valid_all_auc': os.path.join(args.model_name, 'valid_all_auc.csv'),
            #                    'test_all_aupr': os.path.join(args.model_name, 'test_all_aupr.csv'),
            #                    'test_all_auc': os.path.join(args.model_name, 'test_all_auc.csv')}
            #
            # f = open(self.file_names['train'], 'w+'); f.close()
            # f = open(self.file_names['valid'], 'w+'); f.close()
            # f = open(self.file_names['test'], 'w+'); f.close()
            # f = open(self.file_names['valid_all_aupr'], 'w+'); f.close()
            # f = open(self.file_names['valid_all_auc'], 'w+'); f.close()
            # f = open(self.file_names['test_all_aupr'], 'w+'); f.close()
            # f = open(self.file_names['test_all_auc'], 'w+'); f.close()
            # os.utime(args.model_name, None)
        
        self.best_valid = {'mAP': 0,
                           'ACC': 0,
                           'HA': 0,
                           'ebF1': 0,
                           'OF1': 0,
                           'CF1': 0}

        self.best_test = {'mAP': 0,
                          'ACC': 0,
                          'HA': 0,
                          'ebF1': 0,
                          'OF1': 0,
                          'CF1': 0}

    def evaluate(self,
                 valid_metrics,
                 test_metrics,
                 epoch,
                 model,
                 args):

        save_dict_epoch = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'valid_mAP': valid_metrics['mAP'],
            'test_mAP': test_metrics['mAP']
        }
        torch.save(save_dict_epoch,
                   args.model_name+'/'+'epochs'+'/'+'model_epoch'+str(epoch)+'_mAP'+str(format(valid_metrics['mAP'], '.2f'))+'.pt')

        if valid_metrics['mAP'] >= self.best_mAP:
            self.best_mAP = valid_metrics['mAP']
            self.best_test['epoch'] = epoch

            for metric in valid_metrics.keys():
                if not 'all' in metric and not 'time' in metric:
                    self.best_valid[metric] = valid_metrics[metric]
                    self.best_test[metric] = test_metrics[metric]

            print('> Saving Model\n')
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'valid_mAP': valid_metrics['mAP'],
                'test_mAP': test_metrics['mAP']
            }
            torch.save(save_dict, args.model_name + '/best_model.pt')

            print('\n')
            print('**********************************')
            print('best mAP:  {:0.1f}'.format(self.best_test['mAP']*100))
            print('best CF1:  {:0.1f}'.format(self.best_test['CF1']*100))
            print('best OF1:  {:0.1f}'.format(self.best_test['OF1']*100))
            print('**********************************')

        return self.best_valid, self.best_test
