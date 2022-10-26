import argparse
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from models.utils import custom_replace, intraClass_Sim, interClass_Sim
from load_data import get_random_neg_samples


def run_epoch(args, model, data, optimizer, desc, train=False):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset), args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset), args.num_labels).cpu()

    max_samples = args.max_samples

    batch_idx = 0

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):
        if batch_idx == max_samples:
            break

        labels = batch['labels'].float()
        images = batch['image'].float()

        if train:
            images_healthy, labels_healthy = get_random_neg_samples(num=args.batch_size)

            labels_healthy = labels_healthy.float()
            images_healthy = images_healthy.float()

            pred, pred2, label_embeddings = model(images.cuda())
            pred_healthy, pred2_healthy, label_embeddings_healthy = model(images_healthy.cuda())
        else:
            with torch.no_grad():
                pred, pred2, label_embeddings = model(images.cuda())

        if train:
            # print('loss...')
            loss1 = F.binary_cross_entropy_with_logits(pred.view(labels.size(0), -1), labels.cuda(), reduction='none')
            loss2 = F.binary_cross_entropy_with_logits(pred2.view(labels.size(0), -1), labels.cuda(), reduction='none')
            loss_sick = loss1 + loss2

            loss1_healthy = F.binary_cross_entropy_with_logits(pred_healthy.view(labels_healthy.size(0), -1), labels_healthy.cuda(), reduction='none')
            loss2_healthy = F.binary_cross_entropy_with_logits(pred2_healthy.view(labels_healthy.size(0), -1), labels_healthy.cuda(), reduction='none')
            loss_healthy = loss1_healthy + loss2_healthy

            intra_same, intra_diff = intraClass_Sim(label_embeddings, labels.cuda(), healthy=False)
            intra_loss_healthy = intraClass_Sim(label_embeddings_healthy, labels_healthy.cuda(), healthy=True)
            inter_loss = interClass_Sim(label_embeddings, label_embeddings_healthy, labels.cuda())

            con_loss = inter_loss + (intra_same + intra_loss_healthy) / (intra_same + intra_loss_healthy + intra_diff)
            loss_out = loss_sick.mean() + loss_healthy.mean() + 0.7*con_loss

            # print('backward...')
            loss_out.backward()
            # Grad Accumulation
            if (batch_idx + 1) % args.grad_ac_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        start_idx, end_idx = (batch_idx*data.batch_size), ((batch_idx+1)*data.batch_size)

        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0), -1)

        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        batch_idx += 1

    return all_predictions, all_targets


