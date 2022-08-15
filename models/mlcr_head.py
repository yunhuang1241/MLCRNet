import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmseg.ops import resize
from mmseg.models.losses import accuracy
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models import SEGMENTORS
from mmcv.cnn import ConvModule


@SEGMENTORS.register_module()
class MLCRHead(BaseDecodeHead):
    """Multi-Level Context Refinement Networks.
       Args:
           feature_strides (tuple[int]): The strides for input feature maps.
               stack_lateral. All strides suppose to be power of 2. The first
               one is of largest resolution.
       """

    def __init__(self, feature_strides, **kwargs):
        super(MLCRHead, self).__init__(**kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.feature_strides = feature_strides

        context_refine = []
        cls_heads=[]
        self.num_stages = len(self.in_channels)

        for i in range(self.num_stages):
            if i+1 == self.num_stages:
                cls_head = [ConvModule(
                4*self.channels,
                self.channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),]
            else:
                cls_head = [ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                    nn.Dropout2d(self.dropout_ratio)]
            cls_head.append(nn.Conv2d(self.channels,
                                      self.num_classes,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True))
            cls_heads.append(nn.Sequential(*cls_head))


        #squeeze channels
        context_refine.append(
            ConvModule(
                self.in_channels[-1],
                self.channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )

        for i in range(1,len(feature_strides)):
            context_refine.append(
                    MLCR(
                        in_dim=self.channels,
                        low_level_dim=self.in_channels[-(i+1)],
                        out_dim = self.channels,
                        n_class=self.num_classes,
                        up_sample= True
                    )
            )
        self.context_refine = nn.ModuleList(context_refine)
        self.cls_head= nn.ModuleList(cls_heads)

    def forward(self, inputs):
        f = self._transform_inputs(inputs)
        fpn_feature_list = []
        outputs=[]
        for i in range(self.num_stages):
            if i == 0:
                x = self.context_refine[i](f[self.num_stages - i - 1])
            else:
                x = self.context_refine[i](fpn_feature_list[i-1],
                                           f[self.num_stages - i - 1],
                                           outputs[i-1])
            fpn_feature_list.append(x)
            if i == (self.num_stages-1):
                fusion_list = []
                for i in range(0, len(fpn_feature_list)):
                    fusion_list.append(nn.functional.interpolate(
                        fpn_feature_list[i],
                        f[0].shape[2:],
                        mode='bilinear', align_corners=False))
                outputs.append(self.cls_head[i](torch.cat(fusion_list, 1)))
            else:
                outputs.append(self.cls_head[i](x))

        return outputs

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only the last seg map is used."""
        return self.forward(inputs)[-1]

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        loss = dict()

        for i in range(len(seg_logit)):
            if i == (len(seg_logit) - 1):
                compute_accu = True
            else:
                compute_accu = False
            tmp = self.loss(seg_logit[i], seg_label,compute_accu)
            for k,v in tmp.items():
                if k in loss:
                    loss[k] += v
                else:
                    loss[k] = v
        #print(loss)
        return loss


    @force_fp32(apply_to=('seg_logit',))
    def loss(self, seg_logit, seg_label,compute_accu = False):
        """Compute segmentation loss."""
        loss = dict()
        #print(seg_logit.shape)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        if compute_accu:
            loss['acc_seg'] = accuracy(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

class MLCR(nn.Module):
    def __init__(self,in_dim,low_level_dim,out_dim,n_class,up_sample=False):
        super(MLCR, self).__init__()
        self.n_class = n_class
        self.low_level_dim = low_level_dim
        self.up_sample = up_sample

        self.squeeze_l = ConvBNReLU(low_level_dim,in_dim,1)
        self.squeeze_h = ConvBNReLU(in_dim,in_dim,1)

        self.s_branch = SGM(n_class=n_class)
        self.l_branch = ConvBNReLU(2*in_dim,n_class,kernel_size=3,stride=1,padding=1)
        self.g_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim,in_dim,kernel_size=1),
            nn.LayerNorm([in_dim,1,1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,in_dim,kernel_size=1)
        )

        # self.out = ConvBNReLU(in_dim,out_dim,1,1)

    def forward(self,h_feature,l_feature,prob):
        b,c,h,w = h_feature.shape

        l_feature = self.squeeze_l(l_feature)
        h_feature = self.squeeze_h(h_feature)

        cat_feature = torch.cat((l_feature,h_feature),dim =1 )              #(2C,H,W)

        #relation modeling
        local_relation = self.l_branch(cat_feature)                         #L (K,H,W)
        global_relation = self.g_branch(h_feature)                          #G (C,1,1)
        semantic_relation=self.s_branch(h_feature,prob)                     #S  (C,K,1)

        gs_relation = global_relation * semantic_relation                   #GS (C,K,1)

        local_relation = local_relation.view(b,self.n_class,-1)             #BKN
        gs_relation = gs_relation.view(b,c,-1)

        attention = torch.matmul(gs_relation,local_relation).view(b,c,h,w)#LGS (C,H,W)

        #weighting
        o_feature =  h_feature * attention + l_feature

        if self.up_sample:
            o_feature = F.interpolate(o_feature, scale_factor=2, mode='bilinear', align_corners=True)
        return o_feature

class SGM(nn.Module):
    """
        Semantic-Level Context Gather Module:
        Aggregate the semantic-level context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self,n_class = 2,scale = 1):
        super(SGM, self).__init__()
        self.n_class = n_class
        self.scale = scale

    def forward(self,features,seg_logit):
        b,c,h,w = features.shape

        seg_logit = seg_logit.view(b,seg_logit.shape[1],-1)#b c hw
        features = features.view(b,c,-1).permute(0,2,1)#b hw c

        seg_logit = torch.softmax(self.scale * seg_logit,dim=2) # b k hw

        semantic_realtion = torch.matmul(seg_logit ,features).permute(0,2,1).unsqueeze(3)#b c k 1
        return semantic_realtion

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3,stride = 1,padding = 0,bias = False):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                padding=padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


