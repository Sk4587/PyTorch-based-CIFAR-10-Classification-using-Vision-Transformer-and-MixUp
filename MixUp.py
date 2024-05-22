import torch
import numpy as np

import torch
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(15)

class MixUp:
    def __init__(self, alpha, sampling_method):
        self.alpha = alpha
        self.sampling_method = sampling_method

    def mix(self, x, y):

        if(self.sampling_method==1):
            l = np.random.beta(self.alpha,self.alpha)
        if(self.sampling_method==2):
            l = np.random.uniform(0,0.5)

        indices = torch.randperm(x.shape[0]).to(x.device)
        mixed_x = l*x + (1 - l)*x[indices,:,:,:]
        mixed_y = l*y + (1 - l)*y[indices]

        return mixed_x, mixed_y