import torch
import os

from torch.autograd import Variable
from torchvision import datasets, transforms

from model import CapsNet
from smallNorb import SmallNORB
from tools import *

from tqdm import tqdm

test_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (0.3081,))
    ])

test_dataset = SmallNORB('./datasets/smallNORB/', train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

capsnet = CapsNet(reconstruction_type='FC',
        imsize=32,
        num_classes=5,
        routing_iterations = 3,
        primary_caps_gridsize=8,
        num_primary_capsules=32,
        batchnorm=False,
        loss = 'L2',
        leaky_routing=False)

capsnet = capsnet.cuda()
checkpoint = torch.load('./checkpoint/final2.pt')
capsnet.load_state_dict(checkpoint['net'])

check = checkpoint['net']

for key in check:
    print(key)
    path = os.path.join('/workspace/CAPSULE/1/bin', key) + '.bin'
    with open(path, 'wb') as f:
        f.write(check[key].cpu().numpy())

