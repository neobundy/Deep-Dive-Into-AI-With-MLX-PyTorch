import torch
import torch.nn.functional as F

scores = torch.tensor([3.0, 1.0, 0.2])
softmax_scores = F.softmax(scores, dim=0)

print(softmax_scores)
print(sum(softmax_scores))