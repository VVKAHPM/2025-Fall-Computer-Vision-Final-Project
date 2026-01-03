import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def run_clip_resnet_diagnostic(image_path):
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 正在使用设备: {device}")

    # 1. 加载官方 RN50 (ResNet-50) 模型
    # 确保满足助教 "Convolutional Backbone" 的要求
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    # 2. 准备图片和查询语句
    # 建议加入更多对比项，能更清晰地看出模型是在“看方位”还是“数零件”
    try:
        raw_image = Image.open(image_path)
    except Exception as e:
        print(f"错误: 无法打开图片，请检查路径。{e}")
        return

    image_input = preprocess(raw_image).unsqueeze(0).to(device)
    
    test_queries = [
        "the red apple is on the left of",      # [事实真]
        "the green apple is on the left"
        # "two apples higher than the board",       # [事实真]
        # "two apples under the board",     # [数量错]
        # "three apples in total",         # [总数真]
        # "a wooden desk",
        # "one red apple on the top of the desk",
        # "a cat on the floor"             # [无关项]
    ]
    text_input = clip.tokenize(test_queries).to(device)

    # 3. 推理
    with torch.no_grad():
        # 计算相似度得分 (Logits)
        # logits_per_image 的数值即为绝对匹配强度
        logits_per_image, _ = model(image_input, text_input)
        
        # 计算相对概率 (Softmax)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        logits = logits_per_image.cpu().numpy()[0]

    # 4. 打印格式化结果
    print("\n" + "="*75)
    print(f"{'查询文本 (Query)':<30} | {'绝对得分 (Logit)':<15} | {'相对置信度'}")
    print("-" * 75)
    
    for i in range(len(test_queries)):
        print(f"{test_queries[i]:<30} | {logits[i]:<15.2f} | {probs[i]:.2%}")
    print("="*75)

    # 5. 机理深度分析建议
    best_idx = np.argmax(probs)
    print(f"\n[初步诊断结论]:")
    print(f"模型‘最相信’的描述是: '{test_queries[best_idx]}'")
    

if __name__ == "__main__":
    # 请确保路径正确
    run_clip_resnet_diagnostic("images/apple_red_green.jpg")