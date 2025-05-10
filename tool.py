import torch
from torchsummary import summary
import torchvision.models.segmentation.deeplabv3

if __name__ == '__main__':
    # 训练好的模型的路径
    model_path = ''

    from nets.method_ghpa_gab_ca_res import self_net

    #from nets.MALUNet import MALUNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = self_net().to(device).eval()
    #model = MALUNet(num_classes=4).to(device)
    input_size = (3,224,224)
    summary(model, input_size)



