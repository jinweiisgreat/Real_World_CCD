import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn import functional as F


class DistillLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(DistillLoss, self).__init__()
        # 这个公式没有用到temperature
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.anchor_count = 1

    def forward(self, l_txt_feats, txt_feats_gt, labels=None, mask=None):
        distill_loss = self.get_loss_by_feats(l_txt_feats.float().cuda(), txt_feats_gt.float().cuda(), labels)
        return distill_loss

    def get_loss_by_feats(self, l_txt_feats, txt_feats_gt, labels):
        device = torch.device('cuda')

        # 平方项
        square = (l_txt_feats - txt_feats_gt) @ (l_txt_feats - txt_feats_gt).T

        # 相似性
        anchor_dot = (txt_feats_gt @ l_txt_feats.T)  # 这里不除以温度系数（公式中没有温度系数）

        mask_eye = torch.eye(anchor_dot.shape[0], dtype=torch.float32).to(device)
        # 生成只有对角线上的为1的矩阵，保证自己文本与自己的伪文本比较（用于分子）

        mask = torch.eq(labels, labels.T).float().to(device)  # [l_txt_feats.shape[0],l_txt_feats.shape[0]]

        index = torch.arange(l_txt_feats.shape[0] * self.anchor_count).view(-1, 1).to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            index,
            0
        )  # [l_txt_feats.shape[0],l_txt_feats.shape[0]]

        # *logits_mask表示不和自己比（去掉对角线的1），logits_mask只有对角线为0，其他为1
        # mask原来有1有0，对角线肯定为1，其他的如果有相同类，也为1，其他为0
        # mask = mask*logits_mask则把mask原来对角线的去掉，相当于自己不和自己比，和其他同类的会比
        '''
        一开始txt_feats_gt @ l_txt_feats.T 时，我们计算了每一个样本与所有其他样本的相似度
        然后根据分子和分母，从这所有相似度里挑选我们要用的
        而mask或者logits_mask，就是起到在tensor里面进行挑选的作用
        与mask相乘，为1的留下，为0的去掉（该位置在结果里变为0）
        一定要在sum(1)前一刻乘以mask（不能早不能晚），不然会导致各种问题
        sum()之后，就没有挑选哪些元素的概念了
        '''

        mask = mask * logits_mask  # 去掉自己，不和自己比
        # [l_txt_feats.shape[0],l_txt_feats.shape[0]]

        # 解决溢出问题
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()

        # logits = logits*mask_eye#只和自己比
        exp_logits = torch.exp(logits) * logits_mask  # 不和自己比，其他不变

        # exp_logits是不和自己比的，logits是只和自己比的
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [l_txt_feats.shape[0],l_txt_feats.shape[0]]

        mean_log_prob_pos = ((log_prob + square) * mask_eye).sum(1)

        # 如果分子分母的情况一样，可以前面不去*mask,，最后这里再(mask*log_prob).sum(1)
        # 如果分子分母情况不同，那么分开*mask或者logits_mask

        # loss
        loss = - mean_log_prob_pos

        loss = loss.view(self.anchor_count, l_txt_feats.shape[0]).mean()
        return loss


class AlignLoss(torch.nn.Module):
    '''
    用于unlabeldata
    '''

    def __init__(self, temperature=0.01, contrast_mode='all',
                 base_temperature=0.01):
        # GET论文中说明，τa=0.01
        super(AlignLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, img_feats, text_feats, args, labels=None, mask=None):
        device = torch.device('cuda')
        batch_size = img_feats.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 对角线为1，即只将自己作为正样本

        contrast_count = 1

        anchor_count = contrast_count
        self.anchor_count = anchor_count

        img_txt_loss = self.get_loss_by_feats(img_feats, text_feats, args)
        txt_img_loss = self.get_loss_by_feats(text_feats, img_feats, args)

        return img_txt_loss, txt_img_loss

    def get_loss_by_feats(self, a, b, args):
        # a和b会各代入img_feat和txt_feat一次
        anchor_dot = torch.div((a @ b.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()
        '''
        这里对应了softmax(xi)-max(x),解决上溢问题
        logsoftmax解决下溢问题
        '''

        mask_eye = torch.eye(anchor_dot.shape[0], dtype=torch.float32).cuda()
        # 生成只有对角线上的为1的矩阵，保证自己文本与自己的伪文本比较（用于分子）

        exp_logits = torch.exp(logits)
        '''
        algin_loss的分母默认所有，所以不乘logits_mask
        '''
        # 不能一开始就去掉，不然等会减法时会出错

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [batch_size,batch_size]

        '''
        先减再*mask_eye，
        先乘再减，会导致有一部分做减法之后变为负数
        除非你下面再乘一次mask_eye，那上面乘不乘就不影响
        '''

        '''
        注意，这里log_prob*mask_eye，是在挑分子，与挑不挑分母无关
        分母在上面sum的时候已经确定
        如果需要挑，必须在sum之前就挑
        sum之后就已经确定用了哪些元素
        '''
        mean_log_prob_pos = (log_prob * mask_eye).sum(1)  # [batch_size]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [batch_size]
        loss = loss.view(self.anchor_count, a.shape[0]).mean()

        return loss

