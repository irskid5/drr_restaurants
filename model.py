import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState


class DRRAveStateRepresentation(nn.Module):
    def __init__(self, n_items=5, item_features=100, user_features=100):
        super(DRRAveStateRepresentation, self).__init__()
        self.n_items = n_items
        self.random_state = RandomState(1)
        self.item_features = item_features
        self.user_features = user_features

        self.attention_weights = nn.Parameter(torch.from_numpy(0.1 * self.random_state.rand(self.n_items)).float())

    def forward(self, user, items):
        '''
        DRR-AVE State Representation
        :param items: (torch tensor) shape = (n_items x item_features),
                Matrix of items in history buffer
        :param user: (torch tensor) shape = (1 x user_features),
                User embedding
        :return: output: (torch tensor) shape = (3 * item_features)
        '''
        right = items.t() @ self.attention_weights
        middle = user * right
        output = torch.cat((user, middle, right), 0).flatten()
        return output


class Actor(nn.Module):
    def __init__(self, in_features=100, out_features=18):
        super(Actor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear1 = nn.Linear(self.in_features, self.in_features)
        self.linear2 = nn.Linear(self.in_features, self.in_features)
        self.linear3 = nn.Linear(self.in_features, self.out_features)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.tanh(self.linear3(output))
        return output


class Critic(nn.Module):
    def __init__(self, action_size=20, in_features=128, out_features=18):
        super(Critic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.combo_features = in_features + action_size
        self.action_size = action_size

        self.linear1 = nn.Linear(self.in_features, self.in_features)
        self.linear2 = nn.Linear(self.combo_features, self.combo_features)
        self.linear3 = nn.Linear(self.combo_features, self.combo_features)
        self.output_layer = nn.Linear(self.combo_features, self.out_features)

    def forward(self, state, action):
        output = F.relu(self.linear1(state))
        output = torch.cat((action, output), dim=1)
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.output_layer(output)
        return output


class PMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, is_sparse=False, no_cuda=None):
        super(PMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.no_cuda = no_cuda
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()

        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.item_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()

        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.ub.weight.data.uniform_(-.01, .01)
        self.ib.weight.data.uniform_(-.01, .01)

    def forward(self, users_index, items_index):
        user_h1 = self.user_embeddings(users_index)
        item_h1 = self.item_embeddings(items_index)
        R_h = (user_h1 * item_h1).sum(dim=1 if len(user_h1.shape) > 1 else 0) + self.ub(users_index).squeeze() + self.ib(items_index).squeeze()
        return R_h

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds
