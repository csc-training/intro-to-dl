#!/usr/bin/env python
# coding: utf-8

# # Dogs-vs-cats classification with CNNs
#
# In this script, we'll train a convolutional neural network (CNN) to
# classify images of dogs from images of cats using PyTorch.
#
# ## Option 1: Train a small CNN from scratch
#
# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task. However, due to the small number
# of training images, a large network will easily overfit, regardless
# of the data augmentation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from packaging.version import Version as LV
from datetime import datetime
import os
import sys

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert LV(torch.__version__) >= LV("1.0.0")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),             # flatten 2D to 1D
            nn.Linear(17*17*64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).squeeze()


def correct(output, target):
    class_pred = output.round().int()          # set to 0 for <0.5, 1 for >0.5
    correct_ones = class_pred == target.int()  # 1 for correct, 0 for incorrect
    return correct_ones.sum().item()           # count number of correct ones


def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = 0
    num_items = 0

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device).to(torch.float)

        # Do a forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss
        num_batches += 1

        # Count number of correct
        total_correct += correct(output, target)
        num_items += len(target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss/num_batches,
        'accuracy': total_correct/num_items
        }


def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device).to(torch.float)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    return {
        'loss': test_loss/num_batches,
        'accuracy': total_correct/num_items
    }


def log_measures(ret, log, prefix, epoch):
    if log is not None:
        for key, value in ret.items():
            log.add_scalar(prefix + "_" + key, value, epoch)


def main():
    # TensorBoard for logging
    try:
        import tensorboardX
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logdir = os.path.join(os.getcwd(), "logs", "dvc-" + time_str)
        print('TensorBoard log directory:', logdir)
        os.makedirs(logdir)
        log = tensorboardX.SummaryWriter(logdir)
    except ImportError:
        log = None

    num_gpus = torch.cuda.device_count()
    print('Found', num_gpus, 'GPUs')

    # The training dataset consists of 2000 images of dogs and cats, split
    # in half.  In addition, the validation set consists of 1000 images,
    # and the test set of 22000 images.
    #
    # First, we'll resize all training and validation images to a fixed
    # size.
    #
    # Then, to make the most of our limited number of training examples,
    # we'll apply random transformations to them each time we are looping
    # over them. This way, we "augment" our training dataset to contain
    # more data. There are various transformations available in
    # torchvision, see:
    # https://pytorch.org/docs/stable/torchvision/transforms.html

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)
    datapath = os.path.join(datapath, 'dogs-vs-cats/train-2000')

    input_image_size = (150, 150)

    data_transform = transforms.Compose([
            transforms.Resize(input_image_size),
            transforms.RandomAffine(degrees=0, translate=None,
                                    scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    noop_transform = transforms.Compose([
            transforms.Resize(input_image_size),
            transforms.ToTensor()
        ])

    # Data loaders
    batch_size = 25

    print('Train: ', end="")
    train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                         transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    print('Found', len(train_dataset), 'images belonging to',
          len(train_dataset.classes), 'classes')

    print('Validation: ', end="")
    validation_dataset = datasets.ImageFolder(root=datapath+'/validation',
                                              transform=noop_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4)
    print('Found', len(validation_dataset), 'images belonging to',
          len(validation_dataset.classes), 'classes')

    print('Test: ', end="")
    test_dataset = datasets.ImageFolder(root=datapath+'/test',
                                        transform=noop_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)
    print('Found', len(test_dataset), 'images belonging to',
          len(test_dataset.classes), 'classes')

    # Define the network and training parameters
    model = Net()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    print(model)

    num_epochs = 50

    # Training loop
    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, criterion, optimizer)
        log_measures(train_ret, log, "train", epoch)

        val_ret = test(validation_loader, model, criterion)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: "
              f"train loss: {train_ret['loss']:.6f} "
              f"train accuracy: {train_ret['accuracy']:.2%}, "
              f"val accuracy: {val_ret['accuracy']:.2%}")

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    # Inference
    ret = test(test_loader, model, criterion)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")


if __name__ == "__main__":
    main()
