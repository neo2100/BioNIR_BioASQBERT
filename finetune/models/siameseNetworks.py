import torch

class TripletSiamese(torch.nn.Module):

    def __init__(self, baseModel):
        super(TripletSiamese, self).__init__()

        self.net = baseModel

    def forward(self, anchor, positive, negative):

        output1 = self.net(anchor)
        output2 = self.net(positive)
        output3 = self.net(negative)

        return output1, output2, output3

class QuadrupletSiamese(torch.nn.Module):

    def __init__(self, baseModel):
        super(QuadrupletSiamese, self).__init__()
        self.net = baseModel

    def forward(self, anchor, positive, negative1, negative2):

        output1 = self.net(anchor)
        output2 = self.net(positive)
        output3 = self.net(negative1)
        output4 = self.net(negative2)

        return output1, output2, output3, output4