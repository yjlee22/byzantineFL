import random
import copy

import torch

def compromised_clients(args):
    
    max_num = max(int(args.c_frac * args.num_clients), 1)
    
    tmp_idx = [i for i in range(args.num_clients)]
    
    compromised_idxs = random.sample(tmp_idx, max_num)

    return compromised_idxs

def untargeted_attack(w, args):
    mpaf = copy.deepcopy(w)
    for k in w.keys():
        tmp = torch.zeros_like(mpaf[k], dtype = torch.float32).to(args.device)
        w_base = torch.randn_like(mpaf[k], dtype = torch.float32).to(args.device)
        tmp += (w[k].to(args.device) - w_base) * args.mp_lambda
        mpaf[k].copy_(tmp)
    
    return mpaf

