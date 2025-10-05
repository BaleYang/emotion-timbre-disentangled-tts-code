
from collections import OrderedDict
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from text.symbols import symbols


from utils.tools import get_mask_from_lengths, pad


from torch.autograd import Function

    

class MIST(nn.Module):
    def __init__(self, model_config):
        super(MIST, self).__init__()
        self.style_out_dim = model_config['emotion_predictor']['attn_dim']
        self.fc_style = nn.Linear(self.style_out_dim, self.style_out_dim)
        self.fc_enc = nn.Linear(self.style_out_dim, self.style_out_dim)


        self.fc1 = nn.Linear(self.style_out_dim * 2, self.style_out_dim * 2)
        self.fc2 = nn.Linear(self.style_out_dim * 2, self.style_out_dim * 2)
        self.fc3 = nn.Linear(self.style_out_dim * 2, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)


    def forward(self, style_out, encoder_output, src_masks=None):
        x = F.elu(self.fc_style(style_out.squeeze(1)))

        z = F.elu(self.fc_enc(encoder_output))
        
        out = torch.cat((x, z), dim=-1)

        output = F.elu(self.fc1(out))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)

        return output
    