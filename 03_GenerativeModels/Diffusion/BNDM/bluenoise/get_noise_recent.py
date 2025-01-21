import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def noise_padding(noise_small,res=128):
    if res == 128:

        t1,t2,t3,t4 = noise_small[:,0:,...],noise_small[:,1:,...],noise_small[:,2:,...],noise_small[:,3:,...]

        if True:
            top_row = torch.cat((t1,t2),dim=-2)
            bottom_row = torch.cat((t3,t4),dim=-2)