from torch import nn
import torch

# dropout
# x = torch.tensor([1, 3, 5, 7, 9, 11]).float()
# print(x)
# x = nn.Dropout(p=0.5)(x)
# print(x)

# embedding
# x = torch.tensor([1, 2, 4])
# emb = nn.Embedding(10, 10) # num_embeddings: vocab_size
# print(emb(x))

# tok_emb + pos_emb
# x = torch.randn((5, 2, 4))
# y = torch.randn((2, 4))
# print((x + y).shape)

# softmax
# softmax = nn.Softmax(dim=-1)
# x = torch.tensor(
#     [
#         [1, 2, 3, 4, 5],
#         [2, 2, 2, 2, 2]
#     ]
# ).float()
# print(softmax(x))

# apply mask
score = torch.tensor(
    [
        [1, 2, 3],
        [2, 2, 2],
        [4, 5, 6]
    ]
)
mask = torch.tensor(
    [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ]
)
score = score.masked_fill(mask == 0, -10000)
print(score)