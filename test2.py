import torch
from resnet20 import resnet20

model = resnet20().cuda()
saved_state_dict = torch.load('addernet_final.pt')
model.load_state_dict(torch.load('addernet_final.pt'))

model.eval()
