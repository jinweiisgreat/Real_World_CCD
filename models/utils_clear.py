import os
import shutil
from pathlib import Path

# 定义源路径和目标路径 - 请根据你的实际路径修改
source_root = "~/_jinwei/Dataset/CLEAR/CLEAR10/train/labeled_images"
target_root = "~/jinwei/Dataset/CLEAR/CLEAR10_restructured/train"

# 定义映射关系：源文件夹 -> 目标文件夹
folder_mapping = {
    "1": "1", "2": "1", "3": "1", "4": "1",  # 1-4 -> 1
    "5": "2", "6": "2",  # 5-6 -> 2
    "7": "3", "8": "3",  # 7-8 -> 3
    "9": "4", "10": "4"  # 9-10 -> 4
}

# 确保目标目录存在
for i in range(1, 5):
    os.makedirs(os.path.join(target_root, str(i)), exist_ok=True)

# 获取所有可能的类名（从所有源文件夹中）
all_class_names = set()
for src_folder in range(1, 11):
    source_dir = os.path.join(source_root, str(src_folder))
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                all_class_names.add(item)

print(f"找到 {len(all_class_names)} 个唯一类名")

# 处理每个目标文件夹
for target_folder in range(1, 5):
    # 找出要合并到该目标文件夹的源文件夹
    source_folders = [sf for sf, tf in folder_mapping.items() if tf == str(target_folder)]

    print(f"处理目标文件夹 {target_folder}，合并源文件夹 {source_folders}")

    # 对每个类别进行处理
    for class_name in all_class_names:
        # 为该类创建目标目录
        target_class_dir = os.path.join(target_root, str(target_folder), class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # 复制每个源文件夹中的该类别图片
        for source_folder in source_folders:
            source_class_dir = os.path.join(source_root, source_folder, class_name)

            # 检查源目录是否存在
            if os.path.exists(source_class_dir):
                # 获取该类的所有图片
                images = [f for f in os.listdir(source_class_dir)
                          if os.path.isfile(os.path.join(source_class_dir, f))]

                # 复制每张图片到目标目录
                for img in images:
                    source_img_path = os.path.join(source_class_dir, img)
                    # 为避免文件名冲突，添加源文件夹作为前缀
                    target_img_name = f"{source_folder}_{img}"
                    target_img_path = os.path.join(target_class_dir, target_img_name)
                    shutil.copy2(source_img_path, target_img_path)

                print(f"已复制 {len(images)} 张图片从 {source_class_dir} 到 {target_class_dir}")
            else:
                print(f"警告：源目录 {source_class_dir} 不存在")

print("重组完成！")