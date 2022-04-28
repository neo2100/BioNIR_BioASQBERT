# Based on: https://github.com/dimartinot/Text-Semantic-Similarity/blob/master/src/model/losses.py

import torch
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    Ancher and the positive should match while negative should be different than both.
    """

    def __init__(self, epsilon=1.0):
        super(TripletLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, triple):

        euclidean_distance_1 = (triple['anchor'] - triple['positive']).pow(2).sum(-1).sqrt()
        euclidean_distance_2 = (triple['anchor'] - triple['negative1']).pow(2).sum(-1).sqrt()
        #squarred_distance_1 = torch.dot(anchor.view(-1), positive.view(-1))
        #squarred_distance_2 = torch.dot(anchor.view(-1), negative.view(-1))
        
        triplet_loss = F.relu( self.epsilon + euclidean_distance_1 - euclidean_distance_2 ).mean()
        #triplet_loss = torch.clamp(triplet_loss, max=0.25)
        return triplet_loss

class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, epsilon1=1.0, epsilon2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def forward(self, quadruple):

        #squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        #squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        #squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)
        
        #quadruplet_loss = \
        #    F.relu(self.epsilon1 + squarred_distance_pos - squarred_distance_neg) \
        #    + F.relu(self.epsilon2 + squarred_distance_pos - squarred_distance_neg_b)
        
        euclidean_distance_pos = (quadruple['anchor'] - quadruple['positive']).pow(2).sum(-1).sqrt()
        euclidean_distance_neg = (quadruple['anchor'] - quadruple['negative1']).pow(2).sum(-1).sqrt()
        euclidean_distance_neg_b = (quadruple['negative1'] - quadruple['negative2']).pow(2).sum(-1).sqrt()

        quadruplet_loss = \
            F.relu(self.epsilon1 + euclidean_distance_pos - euclidean_distance_neg) \
            + F.relu(self.epsilon2 + euclidean_distance_pos - euclidean_distance_neg_b)

        return quadruplet_loss.mean()