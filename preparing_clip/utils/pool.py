import numpy as np
import torch
from tqdm import tqdm
# class MaxPool(object):
#     def __init__(self, image, kernel_size, stride):
#         self.image = image
#         self.kernel_size = kernel_size
#         self.stride = stride
#     def __call__(self):
#         image_height = self.image.shape[0]
#         image_width = self.image.shape[1]
#         feature_map_height = (image_height - self.kernel_size) // self.stride + 1
#         feature_map_width = (image_width - self.kernel_size) // self.stride + 1
#         feature_map = np.zeros(shape=(feature_map_height, feature_map_width))
#         for i in range(0, feature_map_height):
#             for j in range(0, feature_map_width):
#                 left,top = i*self.stride,j*self.stride
#                 feature_map[i,j] = self._get_max_value(self.image[left : left + self.kernel_size, top : top + self.kernel_size])
#         return feature_map
#     def _get_max_value(self, local_image):
#         return local_image.max()

#用于对image与text进行concat后的tensor进行池化
class MaxPool(object):
    def __init__(self, image, kernel_height, kernel_width, stride):
        self.image = image
        # self.kernel_size = kernel_size
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
    def __call__(self):
        # print(f'image.shape:{self.image.shape}')
        image_height = self.image.shape[0]
        image_width = self.image.shape[1]
        # feature_map_height = (image_height - self.kernel_size) // self.stride + 1
        # feature_map_width = (image_width - self.kernel_size) // self.stride + 1
        feature_map_height = (image_height - self.kernel_height) // self.stride + 1
        feature_map_width = (image_width - self.kernel_width) // self.stride + 1
        feature_map = np.zeros(shape=(feature_map_height, feature_map_width))
        for i in range(0, feature_map_height):
            for j in range(0, feature_map_width):
                left,top = i*self.stride,j*self.stride
                # feature_map[i,j] = self._get_max_value(self.image[left : left + self.kernel_size, top : top + self.kernel_size])
                feature_map[i,j] = self._get_max_value(self.image[left : left + self.kernel_height, top : top + self.kernel_width])
        # return feature_map
        return torch.Tensor(feature_map)
    def _get_max_value(self, local_image):
        return local_image.max()

def MaxPooling(tensor_input=None,kernel_height=None,kernel_width=None,maxpool_stride=1):
    assert kernel_height!=None,kernel_width!=None#必须输入height与width
    # for feat in tensor_input:
        # maxpool=MaxPool(img,kernel_size=maxpool_kernel_size,stride=maxpool_stride)
    maxpool=MaxPool(tensor_input,kernel_height=kernel_height,kernel_width=kernel_width,stride=maxpool_stride)
    maxpool_feat=maxpool()
    # print(maxpool_feat)
    # print(maxpool_feat.shape)
    return maxpool_feat

def MeanPooling(img1):#中值池化，池化后不改变图的大小
    m, n = img1.shape
    img1_ext = cv.copyMakeBorder(img1,1,1,1,1,cv.BORDER_CONSTANT,value=(np.nan,np.nan,np.nan)) / 1.0   #用的是3*3的池化，所以补一圈1 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    img1_ext [np.isnan( img1_ext )]=0
    meanpooling= np.full((m,n),0)
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            pool = [img1_ext[i-1,j-1],img1_ext[i,j-1],img1_ext[i+1,j-1],
                     img1_ext[i-1,j],img1_ext[i,j],img1_ext[i+1,j],
                     img1_ext[i-1,j+1],img1_ext[i,j+1],img1_ext[i+1,j+1]]
            
            meanpooling[i-1,j-1]= np.nanmean(pool) 
    return meanpooling


    
#这个函数用于整理tensor，从[batch_size,n,768]变为[batch_size,n,768]
def modify_tensor(features):
    all_feat = features
    # print(f'all_feat.shape:{all_feat.shape}')
    # all_text_feat = all_text_feat.view(50000,5)

    result = None
    first = True
    # for one_batch_feat in tqdm(all_feat):
    for one_batch_feat in all_feat:
        # output = torch.nn.MaxPool2d([5,1], stride=1, padding=0)
        # print(f'one_batch_feat.shape:{one_batch_feat.shape}')
        # output = MaxPooling(one_batch_feat.cpu(),kernel_height=one_batch_feat.shape[0],kernel_width=1)
        output = MaxPooling(one_batch_feat,kernel_height=one_batch_feat.shape[0],kernel_width=1)
        # print(f'output:{output}')
        # print(f'output.shape:{output.shape}')
        if first == True:
            result = output.unsqueeze(0)#从[1,512]变成[1,1,512]
            first = False
        else:
            result = torch.cat((result,output.unsqueeze(0)), dim=0)
            # result = torch.cat((result.cpu(),output.unsqueeze(0).cpu()), dim=0).cpu()
            # break
    # print(f'result.shape:{result.shape}')
    # result = result.squeeze(1).cpu()
    result = result.squeeze(1)
    # print(f'result.shape:{result.shape}')
    return result
    
    

if __name__ == "__main__": 
    image=np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[4,5,6,7],[4,5,6,7],[4,5,6,7],[4,5,6,7],[4,5,6,7]]])
    # maxpool_kernel_size=2
    kernel_height = 5
    kernel_width = 1
    maxpool_stride=1
    image = torch.Tensor(image)
    print(f'image.shape:{image.shape}')
    # for img in image:
    #     # maxpool=MaxPool(img,kernel_size=maxpool_kernel_size,stride=maxpool_stride)
    #     maxpool=MaxPool(img,kernel_height=kernel_height,kernel_width=kernel_width,stride=maxpool_stride)
    #     maxpool_image=maxpool()
    # print(maxpool_image)
    # print(maxpool_image.shape)
    # MaxPooling(tensor_input=image,kernel_height=kernel_height,kernel_width=kernel_width)
    MaxPooling(tensor_input=image,kernel_height=77,kernel_width=1)


# image
# [[1 2 3 4]
#  [1 2 3 4]
#  [1 2 3 4]
#  [1 2 3 4]]
# maxpool_image
# [[2 4] 
#  [2 4]]