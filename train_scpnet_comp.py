# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name, get_eval_mask, unpack
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings
from utils.np_ioueval import iouEval
import yaml

warnings.filterwarnings("ignore")

import torch.distributed as dist
from mmcv import Config, DictAction
import mmdet
from mmdet3d.datasets import build_dataloader, build_dataset

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

def load_dataset(config='config/bev.py'):
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

def pass_print(*args, **kwargs):
    pass


def train(my_model, optimizer, train_dataset_loader, lovasz_softmax, loss_func, model_save_path, epoch, ignore_label):
    pytorch_device = torch.device('cuda')
    
    loss_list = []
    pbar = tqdm(total=len(train_dataset_loader))
    time.sleep(10)
    # lr_scheduler.step(epoch)
    
    my_model.train()
    for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, train_index, origin_len) in enumerate(train_dataset_loader):
        #train_vox_label: 1x256x256x32
        #train_grid[0]: Nx3 int32 [-40, 40]
        #grain_pt_fea[0]: Nx7 float64
            
        train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
        train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
        point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

        
        # forward + backward + optimize
        outputs = my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0])
        #breakpoint()
        loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label) + loss_func(
            outputs, point_label_tensor)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if i_iter % 100 == 0:
            if len(loss_list) > 0:
                print('epoch %d iter %5d, loss: %.3f\n' %
                        (epoch, i_iter, np.mean(loss_list)))
            else:
                print('loss error')

        optimizer.zero_grad()
        pbar.update(1)
        # global_iter += 1
    pbar.close()

    model_save_name = model_save_path + ('epoch%d.pth' % (epoch))
    torch.save(my_model.state_dict(), model_save_name)
    

def val(my_model, dataset, val_dataset_loader, lovasz_softmax, loss_func, val_batch_size, ignore_label, occ_type):
    pytorch_device = torch.device('cuda')

    my_model.eval()
    with torch.no_grad():
        dataset.occ_eval_metrics = Metric_mIoU(num_classes=18,
                                            use_lidar_mask=True,
                                            use_image_mask=False)

        show_dir = "./pred/baseline_nuscenes/"
        if not os.path.exists(show_dir):
            os.makedirs(show_dir)

        for i_iter_val, (_, val_vox_label, val_grid, _, val_pt_fea, val_index, info) in enumerate(val_dataset_loader):
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            for bat in range(val_batch_size):
                val_label_tensor = val_vox_label[bat,:].type(torch.LongTensor).to(pytorch_device)
                val_label_tensor = torch.unsqueeze(val_label_tensor, 0)
                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                                ignore=ignore_label) + loss_func(predict_labels.detach(), val_label_tensor)

                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                predict_labels = np.squeeze(predict_labels)
                val_vox_label0 = val_vox_label[bat, :].cpu().detach().numpy()
                val_vox_label0 = np.squeeze(val_vox_label0)
                
                #info = dataset.data_infos[val_index[0]]
                if occ_type == 'mini':
                    occ_gt = np.load(f"./data/occ3d-nus-{occ_type}/{info[0]['occ_gt_path']}")
                else:
                    occ_gt = np.load(f"./data/occ3d-nus/{info[0]['occ_gt_path']}")
                
                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
                dataset.occ_eval_metrics.add_batch(predict_labels, val_vox_label0, mask_lidar, mask_camera)   
                
                
                # vis = True
                # if vis:
                #     if val_index[0] % 10 == 0:     
                #         sample_token=info[0]['token']
                #         save_path=os.path.join(show_dir,str(val_index[0]).zfill(4))
                #         np.savez_compressed(save_path,pred=predict_labels,gt=gt_semantics,gt_train=val_vox_label0,sample_token=sample_token)                  
        
        print('Validation per class iou: ')
        del val_vox_label, val_grid, val_pt_fea, val_pt_fea_ten, val_grid_ten, val_label_tensor
        print('\nStarting Evaluation...')
        dataset.occ_eval_metrics.count_miou() 
            

def main(local_rank, args):
    torch.backends.cudnn.benchmark = True
    pytorch_device = torch.device('cuda')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    
    occ_type = "trainval" #mini or trainval
    
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20508")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print
    my_model = model_builder.build(model_config)
    distributed = True
    if distributed:
        #find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True,
            )
    else:
        my_model = my_model.cuda()
        
    
    model_save_path += ''
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print('No existing model, training model from scratch...')

    if not os.path.exists(model_save_path):
        os.makedirs(model_savqe_path)
    print(model_save_path)

    my_model.to(pytorch_device)
    # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    optimizer = optim.AdamW(my_model.parameters(), lr=train_hypers["learning_rate"], weight_decay=0.0001)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, val_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=False,
                                                                  use_multiscan=True,
                                                                  dist=distributed,
                                                                  occ_type=occ_type)
    #train_dataset_loader, val_dataset_loader, val_pt_dataset = tpvformer_data_builder.build()
    
    # training
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    # learning map
    # with open("config/label_mapping/semantic-kitti-all.yaml", 'r') as stream:
    #     semkittiyaml = yaml.safe_load(stream)
    # class_strings = semkittiyaml["labels"]
    # class_inv_remap = semkittiyaml["learning_map_inv"]
    
    #load occ dataset
    dataset = load_dataset()
    
    for epoch in range(train_hypers['max_num_epochs']):
        val(my_model, dataset, val_dataset_loader, lovasz_softmax, loss_func, val_batch_size, ignore_label, occ_type)
        train(my_model, optimizer, train_dataset_loader, lovasz_softmax, loss_func, model_save_path, epoch, ignore_label)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuscene-multiscan.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    #main(args)
