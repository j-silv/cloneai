import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
n_rnn = 512

h1 = torch.zeros(1, batch_size, n_rnn, dtype=torch.float32, device=device)
rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True, device=device)

x1 = torch.ones([batch_size, 65400, n_rnn], device=device)
x2 = torch.ones([batch_size, 65600, n_rnn], device=device)

# prints: True, True, True
print(h1.is_contiguous(), x1.is_contiguous(), x2.is_contiguous())

from pytorch_modelsize import SizeEstimator

se = SizeEstimator(rnn1, input_size=x1.shape)
print(se.estimate_size())
se = SizeEstimator(rnn1, input_size=x2.shape)
print(se.estimate_size())



# runs fine
res, _ = rnn1(x1, h1) 

# fails with:
#  File "venv/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1393, in forward
#   result = _VF.gru(
# RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.
# This error may appear if you passed in a non-contiguous input.
res, _ = rnn1(x2, h1) 

