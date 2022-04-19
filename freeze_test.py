import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, in_features, out_features):
        super(model, self).__init__()
        self.fc1 = nn.Linear(in_features,10)
        self.fc = nn.Linear(10,out_features)
    def forward(self,x):
        x = self.fc1(x)
        return self.fc(x)


net = model(20,2)

predictor = net.cuda()

input = torch.rand([20]).cuda()

gt = torch.ones([2]).cuda()

for i, child in enumerate(net.children()):
    if i == 0:
        for p in child.parameters():
            p.requires_grad=False
# for p in net.fc1.parameters():
#    p.requires_grad=False

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))

loss = nn.MSELoss()
for i in range(200):
    y = predictor(input)
    loss_train = loss(y,gt)

    optim.zero_grad()
    loss_train.backward()
    optim.step()

    #print(loss_train)
    print(y)