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

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        squarred_distance_2 = (anchor - negative).pow(2).sum(1)
        
        triplet_loss = F.relu( self.epsilon + squarred_distance_1 - squarred_distance_2 ).mean()
        
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

    def forward(self, anchor, positive, negative1, negative2):

        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        quadruplet_loss = \
            F.relu(self.epsilon1 + squarred_distance_pos - squarred_distance_neg) \
            + F.relu(self.epsilon2 + squarred_distance_pos - squarred_distance_neg_b)

        return quadruplet_loss.mean()