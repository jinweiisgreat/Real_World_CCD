import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.gpu_utils import check_gpu
from utils.pool import modify_tensor
from utils.misc import multi_apply

# from clip import SimpleTokenizer #上面有_Tokenizer了

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.text_pool_type = 'argmax'
        # self.cast_dtype = self.transformer.get_cast_dtype()
        self.token_embedding = clip_model.token_embedding
        self.vis_dim = 768
        self.ctx_dim = 768

        # self.fc_layer = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        # ])).cuda()
        # self.fc_layer.requires_grad = True#进行学习
        # self.fc_layer.half()
        # self.num_words = 4# a photo of class
        self.num_words = 5  # a photo of class. 多一个句号
        # self.num_words = 15# a photo of class, which is a part of the class of superclass.

        self.ln_learn = nn.Linear(self.vis_dim, self.num_words * self.ctx_dim).cuda()
        self.ln_learn.half()
        self.ln_learn.requires_grad = True
        # self.tokenizer = SimpleTokenizer()
        self.sot_token = _tokenizer.encoder["<|startoftext|>"]
        self.eot_token = _tokenizer.encoder["<|endoftext|>"]

    def forward(self, image_features):
        # print(f'feats.shape:{tokens.shape}')
        # print(f'self.positional_embedding.type(self.dtype).shape:{self.positional_embedding.type(self.dtype).shape}')#[77,768]
        # cast_dtype = self.transformer.get_cast_dtype()
        # text: [b, 77]--->[b, 77, 768] 77个字，每个字的维度为768
        # x = self.token_embedding(text).to(self.cast_dtype)
        # x = self.token_embedding(feats)#不能走这个，毕竟进来就是token

        # x = self.fc_layer(image_features).cuda()
        # psu_feat = self.fc_layer(image_features).cuda()#生成伪造feat
        pseudo_feat = self.ln_learn(image_features).cuda()  # 生成伪造feat，结果是
        # return image_features,pseudo_feat
        pseudo_feat = pseudo_feat.view(-1, self.num_words, self.ctx_dim)  # [batch_size,4,768]
        # psu_feat = psu_feat.unsqueeze(1)#[batch_size,1,768]

        # index = torch.tensor([4])#应该有batch_size个4

        # img_descrip = "A photo of class."

        #         device = pseudo_feat.device
        #         num_preds, num_words, word_dim = pseudo_feat.shape
        #         sot_token = self.token_embedding(torch.tensor([self.sot_token],
        #                                                       device=device))
        #         eot_token = self.token_embedding(torch.tensor([self.eot_token],
        #                                                       device=device))

        #         all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

        # 这里的sot_token和eot_token都已经经过了token_embedding，已经不是整数，而是float了(类似特征)
        # sot_token = sot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)
        # eot_token = eot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)
        # pseudo_tokens = torch.cat([sot_token, pseudo_tokens, eot_token], dim=1)
        # num_words += 2

        context_length = max([seq.shape[0] for seq in pseudo_feat])
        # x, end_token_ids = self.prepare_pseudo_text(pseudo_feat,context_length=context_length + 2)  # add start and stop token
        x, end_token_ids = self.prepare_pseudo_text(pseudo_feat, context_length=77)  # add start and stop token
        # clip_text_features = text_encoder.encode_pseudo_text(pseudo_text, end_token_ids,
        #                                     text_pe=True, normalize=True,
        #                                     return_word_tokens=False)

        # prompts = [prompt_prefix + " " + name + "." for name in classnames]#这里循环了n_cls次
        # prompts = [prompt_prefix + " " + str(i) + "." for i in range(n_cls)]#n_cls用来确认有多少类，就有多少前缀
        # print(f'len(prompts):{len(prompts)}')
        # prompts = [prompt_prefix]#不能直接这样，后面torch.cat()会报错
        #         descriptions = []
        #         for i in range(image_features.shape[0]):
        #             descriptions.append(img_descrip)
        #         # tokenized_descriptions = torch.cat([clip.tokenize(p) for p in descriptions])  # (n_cls, n_tkn)
        #         tokenized_descriptions = torch.cat([clip.tokenize(p, context_length=77, truncate=True).cuda() for p in descriptions])  # (n_cls, n_tkn)
        #         print(f'tokenized_descriptions.shape:{tokenized_descriptions.shape}')#[batch_size,77]

        # text_tokens = clip.tokenize(img_descrip, context_length=77, truncate=True).cuda()#truncate=True表示截断
        # print(f'text_tokens.shape:{text_tokens.shape}')#[batch_size,77]，只有一句，就是[1,77]

        # x = self.token_embedding(tokenized_descriptions).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # image_features[batch_size,768]
        '''
        把image_feat替换"class"这个单词形成的text_feat
        '''
        # print(f'x.shape:{x.shape}')
        # print(f'x:{x}')
        # print(f'psu_feat.shape:{psu_feat.shape}')
        # x = x.index_copy_(1,index.cuda(),psu_feat.cuda())#[batch_size,77,768]
        # return image_features,x
        x = x.type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        # x = x + self.positional_embedding.type(self.dtype).unsqueeze(0)
        # (batch, 1, ctx_dim) + (1, n_ctx, ctx_dim)
        # print(f'x.shape:{x.shape}') #[batch_size,77,768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # x.shape = [batch_size, n_ctx, transformer.width]
        # x,tokens = self.text_global_pool(x,tokens,'argmax')
        # x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        # print(f'x:{x}')
        # x = modify_tensor(x).cuda().type(self.dtype)#不能用这个，梯度会不见
        # x = x[torch.arange(x.shape[0]), x.argmax(dim=-1)]

        '''
        不能用自己那个modify_tensor(x)，除非你是先modify再计算
        先计算的话，会失去原来的梯度
        '''
        # maxpool = nn.MaxPool2d(kernel_size=(77,1),stride=1)
        # x = maxpool(x)#[batch_size,1,768]

        # print(f'x:{x}')
        # print(f'x.shape:{x.shape}')#[batch_size,1,768]
        # x = x.squeeze(1)#从#[batch_size,1,768]变为#[batch_size,768]

        # print(f'x.shape:{x.shape}')#[batch_size,768]
        # x = x.type(self.dtype) @ self.text_projection.type(self.dtype)

        # print(f'x.shape:{x.shape}')
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # print(f'tokenized_descriptions.argmax(dim=-1).shape:{tokenized_descriptions.argmax(dim=-1).shape}')
        # [batch_size]
        # 输出每一个batch的tokenized_descriptions中，“结束符号”的位置，“结束符号”有最大的token值
        # print(f'torch.arange(x.shape[0]):{torch.arange(x.shape[0])}')#一个一维tensor，[0,1,2,3...batch_size-1]
        # x[a,b] a为dim0的下标，b为dim1的下标；
        # 这里的torch.arange(x.shape[0]作为a，代表了batch_size的每一个；argmax作为b则是找到每一句话的结束符号对应的feats所在位置
        # x = x[torch.arange(x.shape[0]), tokenized_descriptions.argmax(dim=-1)] @ self.text_projection
        # print(f'self.text_projection.shape:{self.text_projection.shape}')#[768,768]
        x = x[torch.arange(x.shape[0]), end_token_ids] @ self.text_projection

        return x

        # bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        # ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)

    #     def forward(self, prompts, tokenized_prompts):
    #         x = prompts + self.positional_embedding.type(self.dtype)
    #         x = x.permute(1, 0, 2)  # NLD -> LND
    #         x = self.transformer(x)
    #         x = x.permute(1, 0, 2)  # LND -> NLD
    #         x = self.ln_final(x).type(self.dtype)

    #         # x.shape = [batch_size, n_ctx, transformer.width]
    #         # take features from the eot embedding (eot_token is the highest number in each sequence)
    #         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

    #         return x
    def prepare_pseudo_text(self, pseudo_tokens, context_length):
        device = pseudo_tokens[0].device
        sot_token = self.token_embedding(torch.tensor([self.sot_token],
                                                      device=device))  # [batch_size, n_ctx, d_model]
        eot_token = self.token_embedding(torch.tensor([self.eot_token],
                                                      device=device))
        empty_token = self.token_embedding(torch.tensor([0],
                                                        device=device))
        # 将0经过token_embedding，然后放在后面
        pseudo_tokens = [torch.cat([sot_token, tokens, eot_token], dim=0) for tokens in pseudo_tokens]

        def _pad_sequence(tokens):
            if tokens.shape[0] > context_length:
                # print(f'here1')
                x = tokens[list(range(context_length - 1)) + [tokens.shape[0] - 1]]
                end_token_id = context_length - 1
            else:
                # print(f'here2')
                x = torch.cat([tokens, empty_token.repeat(
                    context_length - tokens.shape[0], 1)], dim=0)
                end_token_id = tokens.shape[0] - 1
            return x, end_token_id

        x, end_token_ids = multi_apply(_pad_sequence, pseudo_tokens)
        x = torch.stack(x, dim=0)

        return x, torch.tensor(end_token_ids, dtype=torch.long, device=x.device)


def get_optim_params(model_name: str):
    if model_name in ['ViT-B/32', 'ViT-B/16']:
        return ['']
    elif model_name in ['ViT-L/14', 'ViT-L/14@336px']:
        return ['visual.transformer.resblocks.23.attn.in_proj_weight',
                'visual.transformer.resblocks.23.attn.in_proj_bias',
                'visual.transformer.resblocks.23.attn.out_proj.weight',
                'visual.transformer.resblocks.23.attn.out_proj.bias',
                'visual.transformer.resblocks.23.ln_1.weight',
                'visual.transformer.resblocks.23.ln_1.bias',
                'visual.transformer.resblocks.23.mlp.c_fc.weight',
                'visual.transformer.resblocks.23.mlp.c_fc.bias',
                'visual.transformer.resblocks.23.mlp.c_proj.weight',
                'visual.transformer.resblocks.23.mlp.c_proj.bias',
                'visual.transformer.resblocks.23.ln_2.weight',
                'visual.transformer.resblocks.23.ln_2.bias',
                'visual.ln_post.weight',
                'visual.ln_post.bias',
                'visual.proj']
    elif model_name in ['ViT-H/14']:
        return ['visual.transformer.resblocks.31.ln_1.weight'
                'visual.transformer.resblocks.31.ln_1.bias'
                'visual.transformer.resblocks.31.attn.in_proj_weight'
                'visual.transformer.resblocks.31.attn.in_proj_bias'
                'visual.transformer.resblocks.31.attn.out_proj.weight'
                'visual.transformer.resblocks.31.attn.out_proj.bias'
                'visual.transformer.resblocks.31.ln_2.weight'
                'visual.transformer.resblocks.31.ln_2.bias'
                'visual.transformer.resblocks.31.mlp.c_fc.weight'
                'visual.transformer.resblocks.31.mlp.c_fc.bias'
                'visual.transformer.resblocks.31.mlp.c_proj.weight'
                'visual.transformer.resblocks.31.mlp.c_proj.bias']
    else:
        print(f"no {model_name}")


class FC(nn.Module):
    def __init__(self, vis_dim, ctx_dim):
        super(FC, self).__init__()
        self.fc_layer = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

    def forward(self, x):
        x = self.fc_layer(x)
        # print(f'x.grad:{x.grad}')
        return x


class CustomCLIP(nn.Module):
    # def __init__(self, cfg, class_num, clip_model):
    def __init__(self, cfg, args, clip_model):
        super().__init__()
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.prompt_learner = PromptLearner(cfg, args, clip_model).cuda()
        # self.prompt_learner = PromptLearner(cfg, class_num, clip_model).cuda()
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.image_encoder = clip_model.visual
        self.model = clip_model
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device('cuda:0')

        optim_params = get_optim_params('ViT-L/14')
        # for name, param in self.model.named_parameters():
        #     if name not in optim_params:
        #         param.requires_grad = False#不在list里面的参数冻结

        # 全部冻结
        for name, param in self.model.named_parameters():
            param.requires_grad = False  # 不在list里面的参数冻结

        # self.prompt_learner.meta_net.requires_grad = True
        vis_dim = self.model.visual.output_dim  # 768
        # ctx_dim = self.model.ln_final.weight.shape[0]#768
        # ctx_dim = 77
        ctx_dim = 768
        # print(f'vis_dim:{vis_dim}')
        # print(f'ctx_dim:{ctx_dim}')
        # self.fc_layer = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        # ])).cuda()
        # self.fc_layer = FC(vis_dim,ctx_dim).cuda()

        # self.fc_layer = nn.Linear(vis_dim, ctx_dim).cuda()
        # self.fc_layer.requires_grad = True#进行学习
        # self.fc_layer.half()
        # self.meta_net.float()

    def forward(self, image, label=None):
        # tokenized_prompts = self.tokenized_prompts
        # logit_scale = self.logit_scale.exp()
        # print(f'image.shape1:{image.shape}')
        # print(f'image:{image}')
        image = image.type(self.dtype)

        # print(f'image:{image}')
        # print(f'self.dtype:{self.dtype}')
        # print(f'image.shape2:{image.shape}')
        # image_features = self.image_encoder(image)
        # image_features = self.image_encoder(image)
        # check_gpu()
        # image_features = self.model.encode_image(image.to(self.device)).float()
        # image_features = self.model.encode_image(image.to(self.device))
        image_features = self.model.encode_image(image.cuda())
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # check_gpu()
        # print(f'image_features.shape:{image_features.shape}')#[batch_size,768]
        # check_gpu()

        '''
        直接映射
        '''
        '''
        要先用tokenize变成token[batch_size,77]才能encode_text
        '''
        # 把image_features直接经过两个线性一个Relu变成token（维度对应就行）
        # tokens = self.meta_net(image_features.type(self.dtype).cuda()).cuda()
        # print(f'image_features:{image_features}')

        # tokens = self.fc_layer(image_features).cuda().long()#long()就全变成0了
        # tokens = self.fc_layer(image_features).cuda()#用图像映射为token
        # tokens = self.fc_layer(image_features).cuda()#用图像映射为token
        # [batch_size,768]

        # tokens = self.fc_layer(image.type(torch.int32)).cuda()#用图像映射为token
        # print(f'tokens:{tokens}')
        # tokens = tokens.type(torch.int32)
        # print(f'tokens:{tokens}')
        # print(f'tokens.shape:{tokens}')
        # print(f'tokens.shape:{tokens.shape}')
        # text_features = self.text_encoder(tokens)
        text_features = self.text_encoder(image_features)
        # text_features = tokens
        # text_features = self.model.encode_text(tokens)
        # print(f'text_features:{text_features}')

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

        '''
        下面这些没用了
        '''


#         prompts = self.prompt_learner(image_features)
#         # prompts = self.prompt_learner(image_features).float()#会报错，prompt估计用的是half
#         # check_gpu()
#         # print(f'prompts.shape:{prompts.shape}')#([batch_size, n_cls, 77, 768]),n_ctx=77
#         # print(f'tokenized_prompts.shape:{tokenized_prompts.shape}')#([n_cls, 77])

#         # text_features = self.model.encode_text(prompts).float()#不能那么写，因为prompt的结构不是[batch_size,句子数目,768]
#         # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         logits = []
#         all_text_features = []
#         count = 0
#         for pts_i, imf_i in zip(prompts, image_features):
#             # print(f'pts_i.shape:{pts_i.shape}')#[n_cls, 77, 768])，batch_size中每个样本一次循环
#             # print(f'imf_i.shape:{imf_i.shape}')#[768])
#             # count += 1
#             # if count >= 5:
#             #     break

#             text_features = self.text_encoder(pts_i, tokenized_prompts)#[n_cls,768]
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             # print(f'text_features.shape:{text_features.shape}')#[n_cls,768]
#             # print(f'logit_scale.shape:{logit_scale.shape}')
#             l_i = logit_scale * imf_i @ text_features.t()#t为转置，[768]*[768,n_cls]=[n_cls]

#             logits.append(l_i)
#             # print(f'l_i.shape:{l_i.shape}')#结果为[n_cls]
#             # check_gpu()
#             all_text_features.append(text_features)
#         logits = torch.stack(logits)
#         all_text_features = torch.stack(all_text_features)
#         # print(f'all_text_features.shape:{all_text_features.shape}')#[batch_size,n_cls,768]
#         all_text_features = all_text_features.squeeze(1)
#         # print(f'all_text_features.shape:{all_text_features.shape}')#[bathc_size,768]
#         # print(f'image_features.shape:{image_features.shape}')#[bathc_size,768]
#         # print(f'logits.shape:{logits.shape}')#[batch_size,n_cls]

#         # print(f'all_text_features.t():{all_text_features.t()}')
#         # temp_logits = image_features @ all_text_features.t()
#         # if label != None:
#         #     label = label.cuda()
#         #     if self.prompt_learner.training:
#         #         return F.cross_entropy(logits, label)
#         #     #F.cross_entropy函数与nn.CrossEntropyLoss类是相似的，但前者更适合于控制更多的细节，并且不需要像后者一样在前面添加一个Softmax层
#         # return logits
#         #原代码如果是无标签，则返回logits，有标签，返回alignloss

#         #这里改成直接返回两种features
#         return image_features,all_text_features


# t为转置，[1,768]*[768,1]=[1,1],这表示1个和另一个对比，要对比batch_size次，所以是[1,batch_size]，这是一个与其他所有元素的比对结果
# 一共有batch_size个元素，因此为[batch_size,batch_size]


def info_nce_logits(features1, features2, args):
    b_ = 0.5 * int(features.size(0))
    # print(f'b_:{b_}')#batch_size

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    # torch.arange(n)，表示创造一个一维tensor，里面是从0到n-1，这里n_views=2，所以是两次0到一半的batch_size-1
    # print(f'labels.shape:{labels.shape}')#torch.Size([2*batch_size])，因为要覆盖到2个view
    # print(f'labels:{labels}')#假如batch_size=128，则为0-128，然后再0-128，两次，加起来torch.Size([2*batch_size])
    # print(f'labels.unsqueeze(0).shape:{labels.unsqueeze(0).shape}')#torch.Size([1,2*batch_size])，即用一个[]把batch_size个数字包起来
    # print(f'labels.unsqueeze(1).shape:{labels.unsqueeze(1).shape}')#torch.Size([2*batch_size,1])，即把每一个数字用[]包起来

    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # 相同为1，不同为0（这里呼应了）
    # print(f'labels.shape:{labels.shape}')#torch.Size([2*batch_size,2*batch_size])
    # unsqueeze(1)后的的每一个元素一个[]，每一个元素都和其他元素相比较，因此这1个元素比较后会有2*batch_size个结果，一共有2*batch_size个元素，因此得到[2*batch_size,2*batch_size]
    # 简单理解为第i个元素的结果，就是第i个[]，这个[]里面有batch_size个1或者0，对应于所有batch_size个元素比较的结果，相同为1，不同为0
    # print(f'labels:{labels}')#相同为1，不同为0（这里呼应了论文loss中的1[n≠i]）
    # 因此，对角线上的肯定为1，但不只有对角线上的为1，因为第i个元素，会在第i和i+128这两个位置重复出现
    # print(f'labels[0][0]:{labels[0][0]}')#为1
    # print(f'labels[0][128]:{labels[0][128]}')#为1
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    '''
    默认2范数，即l2_normalised，如果已经在外面做过了，就不用了
    '''
    # print(f'features.shape:{features.shape}')#[2*batch_size,65536]

    similarity_matrix = torch.matmul(features, features.T)  # 矩阵乘法，两个相乘也可以用@
    '''
    这里对应loss里面的zi*zi'
    '''
    # print(f'similarity_matrix.shape:{similarity_matrix.shape}')#[2*batch_size,2*batch_size]
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    # torch.eye(n，m=None，out=None)生成n行m列的对角线全1，其余部分全0的二维数组，out为数据类型
    # 这里生成了1/2batch_size的方阵，对角线为true
    # print(f'mask.shape:{mask.shape}')#torch.Size([2*batch_size,2*batch_size])
    # print(f'mask:{mask}')#都是True和False
    labels = labels[~mask].view(labels.shape[0], -1)
    '''
    这个view很重要，从[labels.shape[0]]变成[labels.shape[0],1]
    '''
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # ~表示取反，
    '''
    这一步，让similarity_matrix从[2*batch_size,2*batch_size]变为[2*batch_size,2*batch_size-1]
    区别在于去掉了对角线（对应分母的n≠i）
    '''
    # similarity_matrix.shape[0]为batch_size
    # 这里[~mask]相当于把对角线元素去掉了
    # print(f'similarity_matrix.shape:{similarity_matrix.shape}')#[2*batch_size,2*batch_size-1]
    # 因为用mask把对角线去掉了相当于每行都少一个元素，所以列数-1
    # print(f'similarity_matrix:{similarity_matrix}')
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # 找到positives样本，postive样本其实就是自己，以及自己的另一个view
    # print(f'positives.shape:{positives.shape}')#[2*batch_size, 1]
    # print(f'positives:{positives}')#[[数值1],[数值2]，[数值3]...[数值k]]
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    # 找到negatives样本,negatives样本就是自己和另一个view以外的所有样本
    # print(f'negatives.shape:{negatives.shape}')
    # [2*batch_size, 2*batch_size-2]，因为本来similarity_matrix就是[2*batch_size,2*batch_size-1]，现在还有一个1去了positive

    logits = torch.cat([positives, negatives], dim=1)
    # print(f'logits:{logits}')
    # print(f'logits.shape:{logits.shape}')#[2*batch_size,2*batch_size-1]
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    # print(f'labels:{labels}')#全是0，毕竟是torch.zeros
    # print(f'labels.shape:{labels.shape}')#[2*batch_size]

    logits = logits / args.temperature
    # 这里相当于把zi*zi'/τ，算出来了（logits）

    '''
    因为无监督的分母要求n≠i（即自己不能和自己比）
    所以在把positive和negative拼起来之前，把对角线哪一行去掉了（）
    '''

    return logits, labels