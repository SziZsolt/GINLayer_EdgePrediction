import random
import numpy as np
import torch
import torch_geometric as pyg

def set_seed(seed:int):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyg.seed_everything(seed)