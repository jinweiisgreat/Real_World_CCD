import os
import pickle


def txt_to_pkl(txt_path, pkl_save_path):
    # 确保保存路径的目录存在
    pkl_dir = os.path.dirname(pkl_save_path)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
        print(f"创建目录: {pkl_dir}")

    arr = []
    with open(txt_path, 'r') as txt_file:
        for data in txt_file:
            arr.append(data.replace("\n", ""))

    # 使用with语句无需显式关闭文件
    with open(pkl_save_path, "wb") as f:
        pickle.dump(arr, f)

    print(f"成功将 {txt_path} 转换并保存到 {pkl_save_path}")


# 将txt转为pkl
def main():
    dataset_name = 'cifar100'
    txt_to_pkl(
        f'./preparing_clip/txt/{dataset_name}_a_photo_of_label.txt',
        f"./preparing_clip/pkl/{dataset_name}_a_photo_of_label.pkl"
    )


if __name__ == "__main__":
    main()