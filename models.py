import torch
from torchvision.models import resnet50, ResNet50_Weights

import torchvision

import torch.nn as nn

def load_model(name,path=None):
    if name == 'supervised':
        supervised = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        supervised = torch.nn.Sequential(*(list(supervised.children())[:-1]))
        model = nn.Sequential(supervised,
                              nn.Flatten()).to('cuda')
    elif name == 'simclr':

        simclr = torchvision.models.resnet50(pretrained=False)
        simclr.fc = nn.Identity()
        simclr.load_state_dict(torch.load('converted_vissl_simclr1000ep.torch'))
        model = nn.Sequential(simclr,
                                  nn.Flatten()).to('cuda')

    elif name == 'predictor':
        supervised = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        supervised = nn.Sequential(supervised, nn.Softmax(dim=1))
        supervised = supervised.eval()
        supervised = supervised.cuda()
        model = supervised


    elif name == 'simclr_100':

        simclr100 = torchvision.models.resnet50(pretrained=False)
        simclr100.fc = nn.Identity()
        simclr100.load_state_dict(torch.load('converted_vissl_simclr100ep.torch'))
        model = nn.Sequential(simclr100,
                                  nn.Flatten()).to('cuda')
    elif name == 'simclr_200':
        simclr200 = torchvision.models.resnet50(pretrained=False)
        simclr200.fc = nn.Identity()
        simclr200.load_state_dict(torch.load('converted_vissl_simclr200ep.torch'))
        model = nn.Sequential(simclr200,
                                  nn.Flatten()).to('cuda')

    elif name == 'simclr_400':
        simclr400 = torchvision.models.resnet50(pretrained=False)
        simclr400.fc = nn.Identity()
        simclr400.load_state_dict(torch.load('converted_vissl_simclr400ep.torch'))
        model = nn.Sequential(simclr400,
                                  nn.Flatten()).to('cuda')

    elif name == 'simclr_800':
        simclr800 = torchvision.models.resnet50(pretrained=False)
        simclr800.fc = nn.Identity()
        simclr800.load_state_dict(torch.load('converted_vissl_simclr800ep.torch'))
        model = nn.Sequential(simclr800,
                                  nn.Flatten()).to('cuda')
    elif name == 'path':
        pathmodel = torchvision.models.resnet50(pretrained=False)
        pathmodel.fc = nn.Identity()
        pathmodel.load_state_dict(torch.load(path))
        model = nn.Sequential(pathmodel,
                                  nn.Flatten()).to('cuda')



    else:
        raise ValueError('Invalid model name')
    return model

