import random

import torch
from torch import nn


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.hidden_linear = nn.Linear(2, 2, bias=True)
    self.hidden_linear.weight.data.fill_(0.1)
    self.hidden_linear.bias.data.fill_(0.05)
    self.relu = nn.ReLU()
    self.out_linear = nn.Linear(2, 1, bias=True)

  def forward(self, x):
    x = self.hidden_linear(x)
    x = self.relu(x)
    if x[:, 0].sum() < 0.01 or x[:, 1].sum() <0.01:
      print('dead relus')
    x = self.out_linear(x)
    return x


class XOR:
  xor_truth_x = [[1, 1], [0, 0], [1, 0], [0, 1]]
  xor_truth_y = [[0], [0], [1], [1]]


  @classmethod
  def get_xor_batch(cls, batch_size=4):
    sample_x = []
    sample_y = []

    for i in range(batch_size):
      row = random.randint(0, 3)
      sample_x.append(cls.xor_truth_x[row])
      sample_y.append(cls.xor_truth_y[row])

    return torch.Tensor(sample_x), torch.Tensor(sample_y)



if __name__ == '__main__':
  # building the xor function x and y as the dataset

  # fitting the model
  model = Model()
  # model.hidden_linear.weight.data = torch.Tensor([[1, 1], [1, 1]])
  # model.hidden_linear.bias.data = torch.Tensor([0, -1])
  # model.out_linear.weight.data = torch.Tensor([[1, -2]])
  # model.out_linear.bias.data = torch.Tensor([0])
  #
  optim = torch.optim.SGD(model.parameters(), lr=1e-2)
  mse = nn.MSELoss()

  for e in range(10000):
    model.train()
    optim.zero_grad()

    X, Y = XOR.get_xor_batch(4)
    yh = model(X)
    loss = mse(yh, Y)

    loss.backward()
    optim.step()

    if e % 100 == 0:
      print('epoch: %d loss: %.4f' % (e, loss.item()))
  # test the model and its output
  model.eval()
  for s in XOR.xor_truth_x:
    print('input: ', s, ' output: ', model(torch.Tensor(s)))
