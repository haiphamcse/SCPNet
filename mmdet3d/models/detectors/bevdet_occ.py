# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .eqlv2 import EQLv2
from mmcv.runner import force_fp32
import pickle

def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def world2voxel(world, pc_range=[-40, -40, -1.0, 40, 40, 5.4], voxel_size=[0.4, 0.4, 0.4]):
    """
    world: [N, 3]
    """
    # B, N, D, H, W = world.shape[:-1]
    # world_ = world.view(-1, 3)

    return (world - torch.tensor(pc_range)[:3][None, :].to(world.device)) / torch.tensor(voxel_size)[None, :].to(world.device)
    # return voxel_.view(B, N, D, H, W, 3)

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 multi_scale=False,
                 tta=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.multi_scale = multi_scale
        self.tta = tta

        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        if self.multi_scale:
            self.final_conv_ms = nn.ModuleList([
                ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
            for i in range(2)] )

        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

            if self.multi_scale:
                self.predicter_ms = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim*2),
                        nn.Softplus(),
                        nn.Linear(self.out_dim*2, num_classes),
                    )
                for i in range(2)])

        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

        # self.longtail_clsloss = EQLv2()

    def dice_loss_multi_classes(self, preds, voxel_semantics, epsilon=1e-5):
        # assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # convert the feature channel(category channel) as first
        # input = input.permute(1, 0)
        # target = target.permute(1, 0)

        voxel_semantics = voxel_semantics.float()
        # Compute per channel Dice Coefficient
        per_channel_dice = (2 * torch.sum(preds * voxel_semantics, dim=0) + epsilon) / (
            torch.sum(preds * preds, dim=0) + torch.sum(voxel_semantics * voxel_semantics, dim=0) + 1e-4 + epsilon)

        loss = 1.0 - per_channel_dice

        return loss.mean()

    def geo_scal_loss(self, preds, voxel_semantics):
        # pred: N x C
        # ssc_target: N
        # Get softmax probabilities
        n_classes = preds.shape[-1]

        # Compute empty and nonempty probabilities
        empty_probs = preds[:, -1]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = voxel_semantics != 255
        nonempty_target = (voxel_semantics != n_classes-1)
        nonempty_target = nonempty_target[mask].type(preds.dtype)
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )


    def sem_scal_loss(self, preds, voxel_semantics):
        loss = 0
        count = 0
        mask = (voxel_semantics != 255)
        n_classes = preds.shape[-1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = preds[:, i]

            # Remove unknown voxels
            target_ori = voxel_semantics
            p = p[mask]
            target = voxel_semantics[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).type(preds.dtype)
            completion_target_ori[target_ori != i] = 0

            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    # breakpoint()
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count

    def loss_single(self,voxel_semantics,mask_camera,preds, suffix="1x", weight=1.0):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            # mask_camera = mask_camera.to(torch.int32)
            
                
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1).bool()
            num_total_samples = mask_camera.sum()
            # loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            # loss_['loss_occ'] = loss_occ

            #breakpoint()
            preds_ = preds[mask_camera, :]
            voxel_semantics_ = voxel_semantics[mask_camera]
            
            
            
            preds_sm_ = F.softmax(preds_, dim=-1)
            voxel_semantics_onehot_ = F.one_hot(voxel_semantics_, num_classes=18)

            # loss_['loss_occ'] = self.longtail_clsloss(preds_, voxel_semantics_)
            loss_[f'loss_occ_{suffix}'] = self.loss_occ(preds_, voxel_semantics_)
            loss_[f'loss_geo_scale_{suffix}'] = self.geo_scal_loss(preds_sm_, voxel_semantics_)
            loss_[f'loss_sem_scale_{suffix}'] = self.sem_scal_loss(preds_sm_, voxel_semantics_)
            loss_[f'loss_dice_{suffix}'] = self.dice_loss_multi_classes(preds_sm_, voxel_semantics_onehot_)


        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
            
        for k in loss_.keys():
            loss_[k] = loss_[k] * weight
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        # img_feats, _, _, _ = self.extract_img_feat(img, img_metas, **kwargs)
        if self.tta:
            img_feats, _, _, _ = self.extract_img_feat_tta(img, img_metas, **kwargs)

        else:
            img_feats, _, _, _ = self.extract_img_feat(img, img_metas, **kwargs)

        if self.multi_scale:
            occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        else:
            occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)

        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        if self.tta:
            occ_pred[1] = torch.flip(occ_pred[1], dims=(1,))
            # occ_pred[2] = torch.flip(occ_pred[2], dims=(0,))
            # occ_pred[3] = torch.flip(occ_pred[3], dims=(3,4,))
            occ_pred = torch.mean(occ_pred, dim=(0,), keepdim=True)

        # occ_score=occ_pred.softmax(-1)
        # occ_res=occ_score.argmax(-1)

        
        occ_res = occ_pred.argmax(-1)
        #thresholding test
        self.thresholding = True
        if self.thresholding:
            #create filter
            filter = np.asarray(np.where(occ_res.cpu().numpy().reshape(-1) != 17))
            occ_filter = occ_pred.reshape(-1, 18) #nx18
            occ_filter_ = occ_filter[filter, :] #nx18
            occ_filter__ = F.softmax(occ_filter_, dim=-1)

            #filter values by thresholding
            #if max < 0.5 -> assign to 0
            ma, _ = torch.max(occ_filter__, dim=-1)
            occ_arg = torch.where(ma < 0.3, 0, occ_filter__.argmax(-1))
            #occ_arg = occ_filter__.argmax(-1)

            #create new occ
            occ_n = np.full((1, 200, 200, 16), 17)
            occ_n = occ_n.reshape(-1)
            occ_n[filter] = occ_arg.cpu().numpy()
            occ_n = occ_n.reshape(200, 200, 16)

        # if self.ensemble:
        #     pass
        #breakpoint()
        # occ_res = occ_n.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_n]

    @force_fp32()
    def bev_encoder(self, x):
        #print("here")
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if self.multi_scale:
            return x
        elif type(x) in [list, tuple]:
            x = x[0]
        return x


    def get_downsample_mask(self, mask, downsample=1):
        B, D, H, W = mask.shape

        mask = mask.view(B, 
                        D // downsample, downsample, 
                        H // downsample, downsample, 
                        W // downsample, downsample,
                        )
        mask = mask.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        mask = mask.view(-1, downsample ** 3)

        mask = (mask.float().mean(-1) >= 0.5)
        mask = mask.view(B, D // downsample, H //downsample, W // downsample)
        return mask

    def get_downsample_sem(self, sem, downsample=1):
        B, D, H, W = sem.shape

        sem = sem.view(B, 
                        D // downsample, downsample, 
                        H // downsample, downsample, 
                        W // downsample, downsample,
                        )
        sem = sem.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        sem = sem.view(-1, downsample ** 3)

        sem = torch.mode(sem, dim=-1)[0]
        sem = sem.view(B, D // downsample, H //downsample, W // downsample)
        return sem


        # mask = torch.where(gt_depths == 0.0,
        #                             1e5 * torch.ones_like(gt_depths),
        #                             gt_depths)
        # gt_depths = torch.min(gt_depths_tmp, dim=-1).values

        # gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
    

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, depth, frustum_coor, sem2d = self.extract_img_feat(img=img_inputs, img_metas=img_metas, **kwargs)


        # breakpoint()
        losses = dict()
        

        #hai_todo: add multiscale
        if self.multi_scale:
            occ_pred_32 = self.final_conv_ms[1](img_feats[2]).permute(0, 4, 3, 2, 1) # [6, 50, 50, 4, 32]
            occ_pred_16 = self.final_conv_ms[0](img_feats[1]).permute(0, 4, 3, 2, 1) # [6, 100, 100, 8, 32]
            occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        else:
            occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        

        if self.use_predicter:
            if self.multi_scale:
                occ_pred_32 = self.predicter_ms[1](occ_pred_32)
                occ_pred_16 = self.predicter_ms[0](occ_pred_16)
                occ_pred = self.predicter(occ_pred)
            else:
                occ_pred = self.predicter(occ_pred)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.multi_scale:
            # voxel_semantics_16 = F.interpolate(voxel_semantics.float()[:,None,...],scale_factor=0.5, mode='nearest').squeeze(1).contiguous().long()   #6x200x200x16
            # mask_camera_16 = F.interpolate(mask_camera.float()[:,None,...],scale_factor=0.5, mode='nearest').squeeze(1).contiguous().bool()
            # voxel_semantics_32 = F.interpolate(voxel_semantics.float()[:,None,...],scale_factor=0.25, mode='nearest').contiguous().long()  #6x200x200x16
            # mask_camera_32 = F.interpolate(mask_camera.float()[:,None,...],scale_factor=0.25, mode='nearest').squeeze(1).contiguous().bool()

            voxel_semantics_16 = self.get_downsample_sem(voxel_semantics, downsample=2)
            voxel_semantics_32 = self.get_downsample_sem(voxel_semantics, downsample=4)
            mask_camera_16 = self.get_downsample_mask(mask_camera, downsample=2)
            mask_camera_32 = self.get_downsample_mask(mask_camera, downsample=4)

            loss_occ_16 = self.loss_single(voxel_semantics_16, mask_camera_16, occ_pred_16, suffix="2x", weight=0.5)
            loss_occ_32 = self.loss_single(voxel_semantics_32, mask_camera_32, occ_pred_32, suffix="4x", weight=0.25)
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses.update(loss_occ_16)
            losses.update(loss_occ_32)
        else:
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        gt_sem2d, dense_depth_onehot = self.get_sem2d(frustum_coor, voxel_semantics, mask_camera)

        # breakpoint()
        # gt_depth = kwargs['gt_depth']
        # loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        loss_depth = self.img_view_transformer.get_depth_loss(dense_depth_onehot, depth)
        losses['loss_depth'] = loss_depth


            # save_dict = {
            #     'gt_depth': gt_depth.cpu().numpy(),
            #     'gt_dense_depth': dense_depth.cpu().numpy(),
            #     # 'inputs': img_inputs[0][0, 2::3, ...].cpu().numpy(),
            #     # 'gt_depth':results['gt_depth']
            # }

            # with open('results/debug/depth.pickle', 'wb') as handle:
            #     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # quit()

        c_2d, h_2d, w_2d = sem2d.shape[1:4]

        sem2d = sem2d.permute(0, 2, 3, 1).reshape(-1, c_2d)
        gt_sem2d = gt_sem2d.reshape(-1)
        mask_2d = torch.ones((gt_sem2d.shape[0]), dtype=torch.bool, device=gt_sem2d.device)

        loss_2d = self.loss_single(gt_sem2d, mask_2d, sem2d, suffix="2d", weight=0.5)
        losses.update(loss_2d)


        

        return losses


    def get_sem2d(self, frustum_coor, voxel_semantics, mask_camera):
        # voxel_semantics: B, H, W, L
        B, N, D, H, W = frustum_coor.shape[:-1]
        voxel_semantics = voxel_semantics.long()

        frustum_sem = torch.zeros((B, N*D*H*W), dtype=torch.long, device=frustum_coor.device) + 17
        frustum_coor = frustum_coor.view(B, -1, 3)

        pc_range = torch.tensor([-40, -40, -1.0, 40, 40, 5.4])[:3][None, None, :].to(frustum_coor.device)
        voxel_size = torch.tensor([0.4, 0.4, 0.4])[None, None, :].to(frustum_coor.device)

        voxel_frustum_coor = (frustum_coor - pc_range) / voxel_size # B, N, 3
        voxel_frustum_coor = (voxel_frustum_coor).long()
        # voxel_frustum_coor = world2voxel(frustum_coor).long() # B, N, D, H, W, 3

        valid_mask =    (voxel_frustum_coor[..., 0] >= 0) & (voxel_frustum_coor[..., 0] < 200) & \
                        (voxel_frustum_coor[..., 1] >= 0) & (voxel_frustum_coor[..., 1] < 200) & \
                        (voxel_frustum_coor[..., 2] >= 0) & (voxel_frustum_coor[..., 2] < 16) # B, N, 3

        for b in range(B):
            valid_sem_ = voxel_semantics[b, voxel_frustum_coor[b][valid_mask[b]][:, 0], voxel_frustum_coor[b][valid_mask[b]][:, 1], voxel_frustum_coor[b][valid_mask[b]][:, 2]] # n_total
            frustum_sem[b, valid_mask[b]] = valid_sem_

        frustum_sem = frustum_sem.reshape(B, N, D, H, W).permute(0,1,3,4,2).reshape(-1, D) # (B N H W) D
        r_inds, c_inds = torch.nonzero((frustum_sem != 17), as_tuple=True)
        _, unique_r_inds = unique_with_inds(r_inds)

        # breakpoint()
        r_inds = r_inds[unique_r_inds]
        c_inds = c_inds[unique_r_inds]
        
        sem = torch.zeros((B*N*H*W), dtype=torch.long, device=frustum_coor.device) + 17
        sem[r_inds] = frustum_sem[r_inds, c_inds]
        sem = sem.reshape(B, N, H, W)
        # breakpoint()

        reference_depth_frustum = torch.arange(*self.img_view_transformer.grid_config['depth'], dtype=torch.float, device=frustum_coor.device)

        dense_depth_onehot = torch.zeros((B*N*H*W, D), dtype=torch.float, device=frustum_coor.device)
        dense_depth_onehot[r_inds, c_inds] = 1.0
        # dense_depth_onehot = dense_depth_onehot.reshape(B*N, H, W, D)

        # breakpoint()

        return sem, dense_depth_onehot


