import torch
print(torch.__version__)      # 应显示 2.5.1+cu124
print(torch.cuda.is_available()) # 必须显示 True，表示GPU可用