import torch

model_path = 'weights/egeunet_deepsupervision/best_epoch_weights.pth'
model = torch.load(model_path)
state_dict = model.state_dict()
torch.save(state_dict, 'weights/egeunet_deepsupervision/best_epoch_weights_state_dict.pth')