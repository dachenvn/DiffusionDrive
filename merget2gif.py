"""
将 bev_trajectory_0.png 到 bev_trajectory_99.png 合成为GIF动图
依赖：PIL (Pillow) 库
安装依赖：pip install pillow
"""
import os
from pathlib import Path
from PIL import Image

# ========== 配置区（根据实际情况修改） ==========
# 图片所在目录（默认和脚本同目录，若图片在visualization_output下则修改为该路径）
IMAGE_DIR = Path("visualization_output")
# 输出GIF文件名
OUTPUT_GIF = "bev_trajectory_animation.gif"
# GIF帧率（每秒播放的图片数，可调：值越大动图越快）
FPS = 2
# 图片名称前缀（和你的图片命名匹配）
PREFIX = "bev_trajectory_"
# 图片编号范围（0到99）
START_NUM = 0
END_NUM = 99
# ===============================================

def images_to_gif():
    # 1. 初始化图片列表
    image_paths = []
    
    # 2. 按顺序收集图片路径（保证0→99的顺序）
    for num in range(START_NUM, END_NUM + 1):
        img_path = IMAGE_DIR / f"{PREFIX}{num}.png"
        if img_path.exists():
            image_paths.append(img_path)
            print(f"找到图片: {img_path}")
        else:
            print(f"警告：未找到图片 {img_path}，跳过")
    
    # 3. 检查是否有有效图片
    if not image_paths:
        print("错误：未找到任何图片！请检查图片路径和命名是否正确")
        return
    
    # 4. 加载图片并转换为PIL格式
    frames = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            # 转换为RGB（避免透明通道导致GIF异常）
            frames.append(img.convert("RGB"))
        except Exception as e:
            print(f"警告：加载图片 {img_path} 失败，错误：{e}，跳过")
    
    # 5. 保存为GIF
    if frames:
        # 第一个图片作为基础，后续图片作为帧，duration=1000/FPS 是每帧的毫秒数
        frames[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / FPS),
            loop=0,  # 0表示无限循环，1表示只播放1次
            optimize=True  # 优化GIF大小
        )
        print(f"\nGIF生成成功！保存路径：{os.path.abspath(OUTPUT_GIF)}")
        print(f"共包含 {len(frames)} 张图片，帧率：{FPS} FPS")
    else:
        print("错误：没有可加载的有效图片！")

if __name__ == "__main__":
    # 检查依赖
    try:
        import PIL
    except ImportError:
        print("错误：未安装Pillow库，请先执行：pip install pillow")
    else:
        images_to_gif()