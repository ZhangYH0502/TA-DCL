import torch
import torch.nn as nn
import argparse
from load_data import get_data, get_random_neg_samples
from models import CTranModel
from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import os
from tqdm import tqdm
from models.utils import psm_evaluation
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = '0,1'

args = get_args(argparse.ArgumentParser())

print('Labels: {}'.format(args.num_labels))

train_loader, valid_loader, test_loader = get_data(args)
print('train_dataset len:', len(train_loader.dataset))
print('valid_dataset len:', len(valid_loader.dataset))
print('test_dataset len:', len(test_loader))

model = CTranModel(args.num_labels, args.pos_emb, args.layers, args.heads, args.dropout)


def load_saved_model(saved_model_name, model):
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    print("epoch:", checkpoint['epoch'])
    print("valid_mAP:", checkpoint['valid_mAP'])
    return model


print(args.model_name)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.cuda()
# model = load_saved_model(args.model_name + '/best_model.pt', model)
# model = load_saved_model('results/chestmnist.3layer.bsz_64.adam0.0001/epochs-0/model_epoch10_mAP0.16.pt', model)
# print('model loaded')

if args.inference:
    model = load_saved_model(args.saved_model_name, model)
    if test_loader is not None:
        data_loader = test_loader
    else:
        data_loader = valid_loader

    all_preds = torch.zeros(len(data_loader.dataset), args.num_labels).cpu()
    all_targs = torch.zeros(len(data_loader.dataset), args.num_labels).cpu()

    batch_idx = 0
    loss_total = 0

    for batch in tqdm(data_loader, mininterval=0.5, desc='testing', leave=False, ncols=50):

        labels = batch['labels'].float()
        images = batch['image'].float()

        images_healthy, _ = get_random_neg_samples(num=20)
        images_healthy = images_healthy.float()

        with torch.no_grad():
            pred, _, label_embeddings = model(images.cuda())
            _, _, label_embeddings_healthy = model(images_healthy.cuda())

        pred = F.sigmoid(pred)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        pred = 1 - pred
        sim_m = psm_evaluation(label_embeddings, label_embeddings_healthy)
        pred = pred * sim_m

        start_idx, end_idx = (batch_idx * data_loader.batch_size), ((batch_idx + 1) * data_loader.batch_size)
        all_preds[start_idx:end_idx] = pred.data.cpu()
        all_targs[start_idx:end_idx] = labels.data.cpu()
        batch_idx += 1

    test_metrics = evaluate.compute_metrics(args, all_preds, all_targs)

if args.freeze_backbone:
    for p in model.module.backbone.parameters():
        p.requires_grad = False
    for p in model.module.backbone.base_network.layer4.parameters():
        p.requires_grad = True

if args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) #, weight_decay=0.0004)
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

metrics_logger = logger.Logger(args)
loss_logger = logger.LossLogger(args.model_name)

for epoch in range(1, args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch
    ################### Train #################
    all_preds, all_targs = run_epoch(args, model, train_loader, optimizer, 'Training', True)
    # all_preds = torch.where(torch.isnan(all_preds), torch.full_like(all_preds, 0), all_preds)
    print('output train metric:')
    train_metrics = evaluate.compute_metrics(args, all_preds, all_targs)
    loss_logger.log_losses('train.log', epoch, train_metrics)

    ################### Valid #################
    all_preds, all_targs = run_epoch(args, model, valid_loader, None, 'Validating', False)
    print('output valid metric:')
    valid_metrics = evaluate.compute_metrics(args, all_preds, all_targs)
    loss_logger.log_losses('valid.log', epoch, valid_metrics)

    ################### Test #################
    if test_loader is not None:
        all_preds, all_targs = run_epoch(args, model, test_loader, None, 'Testing', False)
        print('output test metric:')
        test_metrics = evaluate.compute_metrics(args, all_preds, all_targs)
    else:
        test_metrics = valid_metrics
    loss_logger.log_losses('test.log', epoch, test_metrics)

    step_scheduler.step(epoch)

    ############## Log and Save ##############
    best_valid, best_test = metrics_logger.evaluate(valid_metrics, test_metrics, epoch, model, args)
    print(args.model_name)
