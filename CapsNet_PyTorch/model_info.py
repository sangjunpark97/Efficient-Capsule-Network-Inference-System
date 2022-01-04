import torch
from model import CapsNet
from torch.autograd import Variable

capsnet = CapsNet(reconstruction_type='FC',
        imsize=32,
        num_classes=5,
        routing_iterations = 3,
        primary_caps_gridsize=8,
        num_primary_capsules=32,
        batchnorm=True,
        loss = 'L2',
        leaky_routing=False)

capsnet = capsnet.cuda()
checkpoint = torch.load('./checkpoint/96.3.pt') 
capsnet.load_state_dict(checkpoint['net'])

print(capsnet)
