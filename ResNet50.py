import torch
import os
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm

for folder in ["results/consistent", "results/inconsistent"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 1. 自动适配数据集的包装器 ---
def get_diagnose_dataloader(batch_size=32):
    # Oxford-IIIT Pet 相比 CIFAR-10 更清晰，适合做 Jigsaw
    
    # 基础预处理：Resize 到 224，这是 ResNet 的标准输入
    dataset = torchvision.datasets.OxfordIIITPet(
        root="./data", 
        download=True, 
        split="test", # 使用测试集
        transform=T.Resize((224, 224)) 
    )
    return dataset

# --- 2. 核心打乱函数 ---
def patch_shuffling_tensor(img_tensor, grid_size=8):
    """直接对 Tensor 进行操作，速度更快"""
    # img_tensor: [C, H, W]
    c, h, w = img_tensor.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patches.append(img_tensor[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w])
    
    import random
    random.shuffle(patches)
    
    # 重新拼接
    rows = []
    for i in range(grid_size):
        rows.append(torch.cat(patches[i*grid_size : (i+1)*grid_size], dim=2))
    shuffled_tensor = torch.cat(rows, dim=1)
    
    return shuffled_tensor

def save_sample(orig_tensor, shuff_tensor, label_name, pred_name, folder, idx):
    # 反标准化 (ImageNet mean/std)
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = T.ToPILImage()
    
    orig_img = to_pil(inv_normalize(orig_tensor[0]).clamp(0, 1))
    shuff_img = to_pil(inv_normalize(shuff_tensor[0]).clamp(0, 1))
    
    # 拼在一起保存：左边原图，右边打乱图
    combined = Image.new('RGB', (448, 224))
    combined.paste(orig_img, (0, 0))
    combined.paste(shuff_img, (224, 0))
    
    filename = f"{folder}/sample_{idx}_{label_name}_to_{pred_name}.jpg"
    combined.save(filename)

# --- 3. 评估程序 ---
def run_mass_diagnosis():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    model = resnet50(weights=weights).to(device)
    model.eval()
    
    dataset = get_diagnose_dataloader()
    
    # 统计数据
    stats = {'correct_orig': 0, 'correct_shuff': 0, 'logit_gain': []}
    
    # 为了公平，我们需要将 Oxford-Pets 的标签映射到 ImageNet 的“猫/狗”大类
    # 简化处理：我们只观察 Top-1 是否依然是“某种猫”或“某种狗”
    
    
    to_tensor = T.ToTensor()
    normalize = weights.transforms()
    count_c, count_i = 0, 0

    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]

         # 原始预测
        orig_t = normalize(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out_orig = model(orig_t)
            pred_orig_idx = out_orig.argmax(1).item()
            label_name = categories[pred_orig_idx].split(',')[0]

        # 打乱预测
        shuff_raw = patch_shuffling_tensor(to_tensor(img), grid_size=4)
        shuff_t = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(shuff_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            out_shuff = model(shuff_t)
            pred_shuff_idx = out_shuff.argmax(1).item()
            pred_name = categories[pred_shuff_idx].split(',')[0]

        # --- 逻辑判断与保存 ---
        if pred_orig_idx == pred_shuff_idx:
            if count_c < 5: # 保存前 5 张一致的
                save_sample(orig_t, shuff_t, label_name, pred_name, "results/consistent", i)
                count_c += 1
        else:
            if count_i < 5: # 保存前 5 张不一致的
                save_sample(orig_t, shuff_t, label_name, pred_name, "results/inconsistent", i)
                count_i += 1
        
        if count_c >= 5 and count_i >= 5:
            break
        
        # # 1. 原始图像预测
        # orig_tensor = normalize(img).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     out_orig = model(orig_tensor)
        #     logit_orig, pred_orig = out_orig.max(1)

        # # 2. 生成打乱图像
        # shuff_tensor_raw = patch_shuffling_tensor(to_tensor(img), grid_size=4)
        # # 注意：打乱后的 Tensor 需要重新 Normalization
        # shuff_tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(shuff_tensor_raw).unsqueeze(0).to(device)
        
        # with torch.no_grad():
        #     out_shuff = model(shuff_tensor)
        #     logit_shuff, pred_shuff = out_shuff.max(1)

        # # 3. 核心统计：如果原始和打乱后的预测结果一致（都是同一种分类），
        # # 且 Logit 没有下降，说明结构确实不重要。
     
        # if pred_orig == pred_shuff:
        #     if count_c < 5: # 保存前 5 张一致的
        #         save_sample(orig_t, shuff_t, label_name, pred_name, "results/consistent", i)
        #         count_c += 1
        # else:
        #     if count_i < 5: # 保存前 5 张不一致的
        #         save_sample(orig_t, shuff_t, label_name, pred_name, "results/inconsistent", i)
        #         count_i += 1

        # if pred_orig == pred_shuff:
        #     stats['correct_shuff'] += 1
        #     stats['logit_gain'].append((logit_shuff - logit_orig).item())

        # stats['correct_orig'] += 1

    # 输出统计结果
    print("\n" + "="*50)
    print(f"预测结果一致样本数: {len(stats['logit_gain'])}/{len(dataset)}")
    print(f"百分比 (Prediction consistency): {len(stats['logit_gain'])/len(dataset):.2%}")
    print(f"平均 Logit 变化 (Confidence Shift): {np.mean(stats['logit_gain']):+.2f}")
    print("="*50)

if __name__ == "__main__":
    run_mass_diagnosis() # 建议跑 200-500 张即可