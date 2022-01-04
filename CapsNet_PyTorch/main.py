import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import CapsNet
from smallNorb import SmallNORB
from tools import *

from tqdm import tqdm


train_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=32./255, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (0.3081,))
    ])

test_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (0.3081,))
    ])

train_dataset = SmallNORB('./datasets/smallNORB/', train=True, download=True, transform=train_transform)
test_dataset = SmallNORB('./datasets/smallNORB/', train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
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

initialize_weights(capsnet)
capsnet = capsnet.cuda()

best_acc = 0
optimizer = torch.optim.Adam(capsnet.parameters(), lr=0.001)

def train(epoch):
    capsnet.train()
    train_correct = 0
    total = 0
    for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch{:3d}".format(epoch)):
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        target = torch.eye(5).cuda().index_select(dim=0, index=target)
        optimizer.zero_grad()
        capsule_output, reconstructions, _ = capsnet(data, target)
        predictions = torch.norm(capsule_output.squeeze(), dim=2)
        loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, 0.0005)

        loss.backward()
        optimizer.step()

        train_correct += (target.max(dim=1)[1] == predictions.max(dim=1)[1]).sum().item()
        total += target.size(0)

    print("acc = {}%".format(train_correct/total))

def test(epoch):
    global best_acc
    capsnet.eval()
    test_correct = 0
    total = 0

    for batch_id, (data, target) in tqdm(list(enumerate(test_loader)), ascii=True, desc="Test {:3d}".format(epoch)):
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        target = torch.eye(5).cuda().index_select(dim=0, index=target)

        capsule_output, reconstructions, predictions = capsnet(data)
        data = denormalize(data)
        loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, 0.0005)

        test_correct += (target.max(dim=1)[1] == predictions.max(dim=1)[1]).sum().item()
        total += target.size(0)

    print("acc = {}%".format(test_correct/total))
    acc = 100.*test_correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': capsnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pt'.format(str(acc)))
        best_acc = acc

for epoch in range(200):
    train(epoch)
    test(epoch)




