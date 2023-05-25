from mmdet3d.datasets import NuScenesDatasetOccpancy
import numpy as np
from mmdet3d.datasets import build_dataloader, build_dataset
from mmcv import Config, DictAction
import mmdet
from mmdet3d.datasets.occ_metrics import Metric_mIoU, Metric_FScore
import os
import torch.nn.functional as F
import torch
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

# def load_dataset():
#     # Data
#     point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
#     # For nuScenes we usually do 10-class detection
#     class_names = [
#         'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#         'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
#     ]
#     multi_adj_frame_id_cfg = (1, 1+1, 1)

#     data_config = {
#     'cams': [
#         'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
#         'CAM_BACK', 'CAM_BACK_RIGHT'
#     ],
#     'Ncams': 6,
#     'input_size': (256, 704),
#     'src_size': (900, 1600),

#     # Augmentation
#     'resize': (-0.06, 0.11),
#     'rot': (-5.4, 5.4),
#     'flip': True,
#     'crop_h': (0.0, 0.0),
#     'resize_test': 0.00,
#     }
#     dataset_type = 'NuScenesDatasetOccpancy'
#     data_root = 'data/nuscenes/'
#     file_client_args = dict(backend='disk')

#     bda_aug_conf = dict(
#         rot_lim=(-0., 0.),
#         scale_lim=(1., 1.),
#         flip_dx_ratio=0.5,
#         flip_dy_ratio=0.5)

#     train_pipeline = [
#         dict(
#             type='PrepareImageInputs',
#             is_train=True,
#             data_config=data_config,
#             sequential=True),
#         dict(type='LoadOccGTFromFile'),
#         dict(
#             type='LoadAnnotationsBEVDepth',
#             bda_aug_conf=bda_aug_conf,
#             classes=class_names,
#             is_train=True),
#         dict(
#             type='LoadPointsFromFile',
#             coord_type='LIDAR',
#             load_dim=5,
#             use_dim=5,
#             file_client_args=file_client_args),
#         # dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
#         # dict(type='OccToMultiViewSem', downsample=1, grid_config=grid_config),
#         dict(type='DefaultFormatBundle3D', class_names=class_names),
#         dict(
#             type='Collect3D', keys=['img_inputs', 'voxel_semantics',
#                                     'mask_lidar','mask_camera'])
#     ]

#     test_pipeline = [
#         dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
#         dict(
#             type='LoadAnnotationsBEVDepth',
#             bda_aug_conf=bda_aug_conf,
#             classes=class_names,
#             is_train=False),
#         dict(
#             type='LoadPointsFromFile',
#             coord_type='LIDAR',
#             load_dim=5,
#             use_dim=5,
#             file_client_args=file_client_args),
#         dict(
#             type='MultiScaleFlipAug3D',
#             img_scale=(1333, 800),
#             pts_scale_ratio=1,
#             flip=False,
#             transforms=[
#                 dict(
#                     type='DefaultFormatBundle3D',
#                     class_names=class_names,
#                     with_label=False),
#                 dict(type='Collect3D', keys=['points', 'img_inputs'])
#             ])
#     ]

#     input_modality = dict(
#         use_lidar=False,
#         use_camera=True,
#         use_radar=False,
#         use_map=False,
#         use_external=False)

#     share_data_config = dict(
#         type=dataset_type,
#         classes=class_names,
#         modality=input_modality,
#         stereo=True,
#         filter_empty_gt=False,
#         img_info_prototype='bevdet4d',
#         multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
#     )

#     test_data_config = dict(
#         pipeline=test_pipeline,
#         ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

#     data = dict(
#         samples_per_gpu=8,
#         workers_per_gpu=12,
#         train=dict(
#             data_root=data_root,
#             ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
#             pipeline=train_pipeline,
#             classes=class_names,
#             test_mode=False,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR'),
#         val=test_data_config,
#         test=test_data_config)

#     for key in ['val', 'train', 'test']:
#         data[key].update(share_data_config)
#     dataset = NuScenesDatasetOccpancy(data)
#     return dataset

def load_dataset(config='configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_scale_dice_ce_sem2d_densedepth.py'):
    cfg = Config.fromfile(config)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)


    cfg.model.pretrained = None


    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    return dataset

def ensemble(bevdet, bevformer):
    occ = F.softmax(torch.tensor(bevdet).float(), dim=-1) + F.softmax(torch.tensor(bevformer).float(), dim=-1)
    occ_pred = np.argmax(occ, axis=-1)
    return occ_pred.cpu().numpy()

def threshold(occ_pred):
    #Done
    #create filter
    occ_res = occ_pred.argmax(-1)
    filter = np.asarray(np.where(occ_res.reshape(-1) == 17))
    #breakpoint()
    occ_filter = occ_pred.reshape(-1, 18) #nx18
    occ_filter_ = occ_filter[filter, :] #nx18
    occ_filter__ = F.softmax(torch.tensor(occ_filter_).float(), dim=-1)

    #filter values by thresholding
    #if max < 0.5 -> assign to 0
    ma, _ = torch.max(occ_filter__, dim=-1)
    occ_arg = torch.where(ma < 0.3, 0, occ_filter__.argmax(-1))
    #occ_arg = occ_filter__.argmax(-1)

    #create new occ
    # occ_n = np.full((1, 200, 200, 16), 17)
    occ_n = occ_res
    occ_n = occ_n.reshape(-1)
    occ_n[filter] = occ_arg.cpu().numpy()
    occ_n = occ_n.reshape(200, 200, 16)
    return occ_n
    
def eval(dataset):
    dataset.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

    print('\nStarting Evaluation...')
    dir = 'logits/bevdet-occ-r50-4d-stereo-24e_scale_dice_ce_sem2d_densedepth/'
    dir_ = 'logits/bevformer_vov_scale_loss_fliphor_sigmoid_eqlv2_dice_masking/'
    arr = os.listdir(dir)
    #breakpoint() #3240
    for index in arr: 
        i = index.replace(".npz", "")
        #breakpoint()
        info = dataset.data_infos[int(i)]
        # a = dataset.data_infos[3240]
        # breakpoint()
        occ_pred = np.load(f'{dir}{index}')['pred']
        occ_pred = threshold(occ_pred)
        # occ_pred = ensemble(np.load(f'{dir}{index}')['pred'], np.load(f'{dir_}{index}')['pred'])
        #occ_pred = np.argmax(occ_pred, axis=-1)
        #breakpoint()
        occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
        gt_semantics = occ_gt['semantics']
        # #check for others
        # item, count = np.unique(gt_semantics, return_counts=True)
        # if 0 in item:
        #     if count[0] > 50:
        #         print(index)
        #         print(item, count)
        
        #breakpoint()
        mask_lidar = occ_gt['mask_lidar'].astype(bool)
        mask_camera = occ_gt['mask_camera'].astype(bool)
        # occ_pred = occ_pred
        dataset.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

    return dataset.occ_eval_metrics.count_miou()  

if __name__ == '__main__':
    dataset = load_dataset()
    eval(dataset)
    #breakpoint()

