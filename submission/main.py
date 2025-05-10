import os
import torch
import numpy as np


class ConfusionMatrix(object):
    def __init__(self, num_classes, ignore_index=0):
        """
        :param num_classes: 类别总数
        :param ignore_index: 要忽略的类别（通常是背景类），默认为 0
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None

    def update(self, a, b):
        """
        更新混淆矩阵
        :param a: 真实标签
        :param b: 预测结果
        """
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            # 更新混淆矩阵
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        """
        重置混淆矩阵
        """
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        """
        计算全局准确率、每个类别的准确率和 IoU
        :return: 全局准确率、每个类别的准确率、每个类别的 IoU
        """
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()  # 全局准确率

        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)

        # 计算 IoU，忽略背景类（类别 0）
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        # 忽略背景类 0 的 IoU 计算
        if self.ignore_index is not None:
            acc = acc[1:]  # 忽略背景类 0 的准确率
            iu = iu[1:]    # 忽略背景类 0 的 IoU

        return acc_global, acc, iu

    def __str__(self):
        """
        打印混淆矩阵的计算结果
        """
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

def count_model_parameters(model):
    """
    计算模型参数总量
    :param model_path: 模型文件路径（.pt或.pth），要求使用 torch.save(model, 'model.pth') 保存模型
    :return: 模型参数总量
    """
    assert model is not None, "模型为空"
        #raise FileNotFoundError(f"模型文件 {model} 不存在")

    # 加载模型
    #model = torch.load(model_path, map_location='cpu')
    model.to(torch.device('cpu'))
    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def seg(pred_dir, gt_dir, num_classes):
    """
    计算指定目录下所有预测文件与真实标签文件的平均 IoU
    :param pred_dir: 预测结果文件夹路径
    :param gt_dir: 真实标签文件夹路径
    :param num_classes: 类别总数
    :return: 每个类别的平均 IoU 数组
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match"

    cm = ConfusionMatrix(num_classes=num_classes)

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        pred_np = np.load(os.path.join(pred_dir, pred_file))
        gt_np = np.load(os.path.join(gt_dir, gt_file))
        
        # 转换为 PyTorch 张量
        pred = torch.from_numpy(pred_np).long()
        gt = torch.from_numpy(gt_np).long()

        # 更新混淆矩阵
        cm.update(gt, pred)
    
    # 打印并返回混淆矩阵的计算结果
    print(cm)
    _, _, mean_ious = cm.compute()

    return mean_ious

if __name__ == "__main__":
    # 训练好的模型的路径
    model_path = 'model.pth'
    from my_nets.GEAUNet import self_net
    model = self_net(n_classes=4).eval()
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict)
    score = 0
    ## 计算模型参数分数 ###
    total_params = count_model_parameters(model)
    norm_params = total_params / 1_000_000
    print(f"模型的参数总量为: {norm_params:.6f} M.")
    score_para = 0
    if norm_params > 17:
        score_para = 10
    else:
        if norm_params < 1:
            score_para = 70
        else:
            score_para = 70 - 15 / 4 * (norm_params - 1)
    print(f"模型参数的分数为 {score_para}")
    score += score_para
    ##################

    ### 计算 class IoU 分数 ####
    pred_dir = 'c_predictions'
    #base_dir = 'results/baseline_predictions/npy'
    base_dir = 'c_predictions'
    #gt_dir = 'results/ground_truths'
    gt_dir = '../gt_files'
    num_classes = 4  # 异常类型数
    improvement_threshold = 0.06

    # 计算 IoU
    print("计算 pre_IoU...")
    pre_IoU = seg(pred_dir, gt_dir, num_classes)
    print("计算 base_IoU...")
    base_IoU = seg(base_dir, gt_dir, num_classes)

    thr = 130
    for cls, (pre, base) in enumerate(zip(pre_IoU, base_IoU), start=1):
        delta = pre - base
        if delta >= improvement_threshold:
            score_class = 100
        else:
            if delta <= 0:
                score_class = 0
            else:
                score_class = 40 + (thr * delta) ** 2
        print(f"Class {cls} 分数：{score_class}")
        score += score_class
    print(f"最终分数：{score}")
