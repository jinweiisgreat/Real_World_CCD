from tqdm import tqdm
from loguru import logger
import clip
import numpy as np
import pickle
import torch


def encode_descrip_txt(file_name):
    model, preprocess = clip.load('/home/ps/_lzj/GCD/Lmodel/ViT-L-14.pt')  # 直接读取本地模型[batch_size,768]
    model.cuda().eval()

    with open(file_name, 'rb') as f:
        text_data = pickle.load(f)
    f.close()

    result = None
    first = True

    for img_descrip in tqdm(text_data):

        # 获取tokens
        text_tokens = clip.tokenize(img_descrip, context_length=77, truncate=True).cuda()  # truncate=True表示截断
        # [batch_size,77]

        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        if first == True:
            result = text_features.unsqueeze(0)
            first = False
        else:
            result = torch.cat((result.cpu(), text_features.unsqueeze(0).cpu()), dim=0)

    print(f'result.shape:{result.shape}')
    # [data数量,1,768]
    # 这个1到后面再squeeze()
    # 因为如果是数据库检索topk，那就是[data数量,topk,768]

    return result


@logger.catch
def main():
    '''
    要读取的pkl位置
    '''
    dataset_name = 'cifar100'
    file_name = f"./preparing_clip/pkl/{dataset_name}_a_photo_of_label.pkl"
    '''
    使用函数获得结果，将pkl中的数组读取，并用CLIP的text encoder提前转换为特征
    '''
    result = encode_descrip_txt(file_name)
    '''
    将结果保存为pth
    '''
    torch.save(result, f"./preparing_clip/pth/{dataset_name}_a_photo_of_label.pth")


if __name__ == "__main__":
    main()
