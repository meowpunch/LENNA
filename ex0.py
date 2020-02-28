import torch
import torchvision
import torchprof

model1 = torchvision.models.alexnet(pretrained=False).cuda()
model2 = torchvision.models.resnet18(pretrained=False).cuda()

x = torch.rand([1, 3, 224, 224]).cuda()


def model():
    model1(x)
    model2(x)


with torchprof.Profile(model1, model2, use_cuda=True) as prof:
    model()
print(prof.display(show_events=False))

help(torchprof.profile.Profile)

