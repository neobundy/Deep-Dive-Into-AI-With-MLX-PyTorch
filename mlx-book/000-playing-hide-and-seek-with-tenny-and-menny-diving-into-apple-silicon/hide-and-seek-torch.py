import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Where are you, Tenny?")
print(f"Tenny: Here I am, in {device}!")