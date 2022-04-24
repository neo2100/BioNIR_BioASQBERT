import torch

class TripletSiamese(torch.nn.Module):

    def __init__(self, baseModel):
        super(TripletSiamese, self).__init__()

        self.net = baseModel

    def forward(self, triple):
        output = {}
        output['anchor'] = self.net(triple['anchor'])
        output['positive'] = self.net(triple['positive'])
        output['negative1'] = self.net(triple['negative1'])

        return output

class QuadrupletSiamese(torch.nn.Module):

    def __init__(self, baseModel):
        super(QuadrupletSiamese, self).__init__()
        self.net = baseModel

    def forward(self, quadruple):
        output = {}
        output['anchor'] = self.net(quadruple['anchor'])
        output['positive'] = self.net(quadruple['positive'])
        output['negative1'] = self.net(quadruple['negative1'])
        output['negative2'] = self.net(quadruple['negative2'])

        return output