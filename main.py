import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
pathname = ""

def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation

    # create dataloader

    # create optimizer

    # create scheduler 

    # ctreat summary writer

        # train

            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients

            # forward

            # backward

            # optimize

        # scheduler adjusts learning rate

        # test
    model.to(device)
    critetion = nn.CrossEntropyLoss()
    transform = transforms.Compose(
    [   
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                        (0.5, 0.5, 0.5))
    ])
    batch_size = 64

    optimizer = optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=15)

    os.makedirs("./run", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    writer = SummaryWriter("./run" + pathname)

    # create dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # create dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  

    for epoch in range(15):

        model.train()
        running_loss = 0.0

        train_loop = tqdm(trainloader, desc=f"epoch: {epoch+1}/{15}", unit="batch")

        for data in train_loop:
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = critetion(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())
        
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        test_loop = tqdm(trainloader, desc=f"Testing Epoch {epoch+1}", unit="batch", leave=False)
        with torch.no_grad():
            for data in test_loop:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        writer.add_scalar("loss/epoch", avg_loss, epoch)
        writer.add_scalar("accuracy/epoch", correct / total * 100, epoch)
        print(f'epoch {epoch+1} finished. avg Loss: {avg_loss:.4f} train Acc: {correct / total * 100:.2f}%')

    torch.save(model.state_dict(), "./models" + pathname + ".pth")
    writer.flush()
    writer.close()

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

def test(model, args):
    '''
    input: 
    model: linear classifier or full-connected neural network classifier
    args: configuration
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
    # forward
    state_dict = torch.load("./models" + pathname + ".pth")
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # create testing dataset

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), 
                        (0.5, 0.5, 0.5)),
    ])
    batch_size = 32

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the model on the test images: {100 * correct / total : .2f}%')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    args = parser.parse_args()


    pathname = "/" + args.model

    # create model
    
    if args.model == 'vgg':
        model = VGG()
    elif args.model == 'resnet':
        model = ResNet()
    elif args.model == 'resnext':
        model = ResNext()

    # train / test

    if args.run == 'train':
        train(model, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError

