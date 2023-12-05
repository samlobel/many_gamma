import torch


asdf = torch.zeros((2, 2, 2))

# print(asdf[[0, 1], [0, 1], [0, 1]].shape)


# print(asdf[:, :, [0, 0, 0]].shape)

batch_size, num_heads, num_actions = asdf.shape
indices = torch.zeros_like(asdf)


# https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

asdf[torch.arange(asdf.shape[0],), :, torch.arange(asdf.shape[2])]