# -*- coding:utf-8 -*-
# author: Xinge
# @file: data_builder.py 

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_tta, collate_fn_BEV_ms, collate_fn_BEV_ms_tta
from dataloader.pc_dataset import get_pc_model_class
from dataloader.dataset_wrapper import custom_collate_fn, DatasetWrapper_NuScenes


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],
          use_tta=False,
          use_multiscan=False,
          use_waymo=False,
          dist=False,
          occ_type="mini"):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version=f'v1.0-{occ_type}', dataroot=data_path, verbose=True)
        print("Using Nuscenes")
        train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                     label_mapping=label_mapping, nusc=nusc)
        val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                                     label_mapping=label_mapping, nusc=nusc)
        # train_pt_dataset = DatasetWrapper_NuScenes(
        #     train_dataset,
        #     grid_size=grid_size,
        #     fixed_volume_space=dataset_config['fixed_volume_space'],
        #     max_volume_space=dataset_config['max_volume_space'],
        #     min_volume_space=dataset_config['min_volume_space'],
        #     fill_label=dataset_config["fill_label"],
        #     phase='train',
        #     scale_rate=scale_rate,
        # )

        # val_pt_dataset = DatasetWrapper_NuScenes(
        #     val_dataset,
        #     grid_size=grid_size,
        #     fixed_volume_space=dataset_config['fixed_volume_space'],
        #     max_volume_space=dataset_config['max_volume_space'],
        #     min_volume_space=dataset_config['min_volume_space'],
        #     fill_label=dataset_config["fill_label"],
        #     phase='val',
        #     scale_rate=scale_rate,
        # )
    else:  
        train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                    return_ref=train_ref, label_mapping=label_mapping, nusc=nusc)
        val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                                return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

    # train_dataset = get_model_class(dataset_config['dataset_type'])(
    #     train_pt_dataset,
    #     grid_size=grid_size,
    #     flip_aug=True,
    #     fixed_volume_space=dataset_config['fixed_volume_space'],
    #     max_volume_space=dataset_config['max_volume_space'],
    #     min_volume_space=dataset_config['min_volume_space'],
    #     ignore_label=dataset_config["ignore_label"],
    #     rotate_aug=True,
    #     scale_aug=True,
    #     transform_aug=True
    # )
    # if use_tta:
    #     val_dataset = get_model_class(dataset_config['dataset_type'])(
    #         val_pt_dataset,
    #         grid_size=grid_size,
    #         flip_aug=True,
    #         fixed_volume_space=dataset_config['fixed_volume_space'],
    #         max_volume_space=dataset_config['max_volume_space'],
    #         min_volume_space=dataset_config['min_volume_space'],
    #         ignore_label=dataset_config["ignore_label"],
    #         rotate_aug=True,
    #         scale_aug=True,
    #         return_test=True,
    #         use_tta=True,
    #     )
    dataAug = 0
    if dataAug:
        train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        rotate_aug=True,
        flip_aug=True,
        ignore_label=dataset_config["ignore_label"],
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        return_test=True,
        )
    else:
        train_dataset = get_model_class(dataset_config['dataset_type'])(
            train_pt_dataset,
            grid_size=grid_size,
            rotate_aug=False,
            flip_aug=False,
            ignore_label=dataset_config["ignore_label"],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            return_test=True,
        )
    if use_tta:
        if dataAug:
            val_dataset = get_model_class(dataset_config['dataset_type'])(
                val_pt_dataset,
                grid_size=grid_size,
                rotate_aug=False,  # True
                flip_aug=False,  # True
                ignore_label=dataset_config["ignore_label"],
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                return_test=True,
            )
        else:
            val_dataset = get_model_class(dataset_config['dataset_type'])(
                val_pt_dataset,
                grid_size=grid_size,
                rotate_aug=False,
                flip_aug=False,
                ignore_label=dataset_config["ignore_label"],
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                return_test=True,
            )
        if use_multiscan:
            collate_fn_BEV_tmp = collate_fn_BEV_ms_tta
        else:
            collate_fn_BEV_tmp = collate_fn_BEV_tta
    else:
        val_dataset = get_model_class(dataset_config['dataset_type'])(
            val_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
            return_test=True,
        )
        if use_multiscan or use_waymo:
            collate_fn_BEV_tmp = collate_fn_BEV_ms
        else:
            collate_fn_BEV_tmp = collate_fn_BEV
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None
        
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV_tmp,
                                                       shuffle= False if dist else train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"],
                                                       sampler=sampler)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV_tmp,
                                                     shuffle= False if dist else val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"],
                                                     sampler=val_sampler)

    if use_tta:
        return train_dataset_loader, val_dataset_loader, val_pt_dataset
    else:
        return train_dataset_loader, val_dataset_loader, val_pt_dataset
