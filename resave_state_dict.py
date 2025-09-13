import torch
from pickle import *    
state_dict = torch.load("trained/addernet_MNIST_best.pt")
state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
with open("addernet_MNIST_best.pickle",'wb') as dout:
    
    saver = Pickler(dout)
    saver.dump(state_dict)