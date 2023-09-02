import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# action_mean = torch.tensor([1,2,3],dtype=float)
# action_mean_batch = torch.tensor([[1,2,3],[4,5,6]],dtype=float)
# action_var = torch.tensor([1,1,1],dtype=float)
# cov_mat = torch.diag(action_var) #strange,why use unsqueeze,batch input?
# cov_mat_ex = action_var.expand_as(action_mean_batch)
# print(cov_mat_ex)
# dist = MultivariateNormal(action_mean_batch, cov_mat)
# print(dist.sample())

actions = [torch.tensor([1,2,3],dtype=float),torch.tensor([4,5,6],dtype=float)]
old_actions  = torch.squeeze(torch.stack(actions, dim=0)).detach()
print(old_actions)