 
import torch
import torch.nn.functional as F
import math
from torch import nn


def custom_replace(tensor, on_neg_1, on_zero, on_one):
    res = tensor.clone()
    res[tensor == -1] = on_neg_1
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# def build_position_encoding(args):
#     N_steps = args.hidden_dim // 2
#     if args.position_embedding in ('v2', 'sine'):
#         # TODO find a better way of exposing other arguments
#         position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
#     elif args.position_embedding in ('v3', 'learned'):
#         position_embedding = PositionEmbeddingLearned(N_steps)
#     else:
#         raise ValueError(f"not supported {args.position_embedding}")
#
#     return position_embedding


def xavier_init(m):
    print('Xavier Init')
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def interClass_Sim(tensor1, tensor2, labels1):

    bsize, seqlen, seqdim = tensor1.size()[0], tensor1.size()[1], tensor1.size()[2]

    tensor1 = tensor1.view(-1, seqdim)
    tensor2 = tensor2.view(-1, seqdim)
    labels1 = labels1.view(bsize*seqlen)

    masks = labels1.ge(0.5)

    cos_sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    cos_sim_diff = cos_sim[masks, :]
    cos_sim_same = cos_sim[~masks, :]

    return cos_sim_same.sum(-1) / (cos_sim_diff.sum(-1)+cos_sim_same.sum(-1))


def intraClass_Sim(tensor, labels, healthy=False):

    bsize, seqlen, seqdim = tensor.size()[0], tensor.size()[1], tensor.size()[2]

    tensor = tensor.view(-1, seqdim)
    labels = labels.view(bsize*seqlen)

    # print(tensor.size())
    # print(labels.size())

    seqlen_r = tensor.size()[0]

    cos_sim_diff = torch.tensor(0).cuda()
    cos_sim_same = torch.tensor(0).cuda()

    if healthy:
        cout_h = 1
        for i in range(seqlen_r):
            for j in range(seqlen_r):
                if i == j:
                    continue
                cos_sim_same = cos_sim_same + F.cosine_similarity(tensor[i, :].unsqueeze(1), tensor[j, :].unsqueeze(1), dim=0)
                cout_h = cout_h + 1
        return cos_sim_same / (cout_h - 1)
    else:
        cout_h = 1
        cout_u = 1
        if labels.sum(-1).item() == 0:
            for i in range(seqlen_r):
                for j in range(seqlen_r):
                    if i == j:
                        continue
                    cos_sim_same = cos_sim_same + F.cosine_similarity(tensor[i, :].unsqueeze(1), tensor[j, :].unsqueeze(1), dim=0)
                    cout_h = cout_h + 1
            return cos_sim_same/(cout_h-1)
        else:
            for i in range(seqlen_r):
                for j in range(seqlen_r):
                    if i == j:
                        continue
                    if labels[i].item() == 0 and labels[j].item() == 0:
                        cos_sim_same = cos_sim_same + F.cosine_similarity(tensor[i, :].unsqueeze(1), tensor[j, :].unsqueeze(1), dim=0)
                        cout_h = cout_h + 1
                    else:
                        cos_sim_diff = cos_sim_diff + F.cosine_similarity(tensor[i, :].unsqueeze(1), tensor[j, :].unsqueeze(1), dim=0)
                        cout_u = cout_u + 1
            return cos_sim_same/(cout_h-1), cos_sim_diff/(cout_u-1)


def psm_evaluation(tensor1, tensor2):
    bsize = tensor2.size()[0]
    sim_m = []
    for i in range(bsize):
        cos_sim = F.cosine_similarity(tensor1[0, :, :], tensor2[i, :, :], dim=1)
        sim_m.append(cos_sim)
    sim_m = torch.stack(sim_m, dim=2)
    sim_m[sim_m < 0.7] = 0
    sim_m[sim_m >= 0.7] = 1
    sim_m = sim_m.sum(-1)
    sim_m[sim_m >= 1] = 1
    return sim_m

