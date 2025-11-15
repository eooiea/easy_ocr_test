import torch
print(torch.__version__) # 버전이 출력되어야 함
print(torch.cuda.is_available()) # GPU를 사용한다면 'True'가 출력되어야 함