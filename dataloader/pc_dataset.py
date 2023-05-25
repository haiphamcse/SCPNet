# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle
import pathlib
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label


def unpack(compressed):  # from samantickitti api
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def get_eval_mask(labels, invalid_voxels):  # from samantickitti api
  """
  Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
  :param labels: input ground truth voxels
  :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
  :return: boolean mask to subsample the voxels to evaluate
  """
  masks = np.ones_like(labels, dtype=np.bool)
  masks[labels == 255] = False
  masks[invalid_voxels == 1] = False

  return masks


from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        ### remap completion label
        remapdict = semkittiyaml['learning_map']
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.comletion_remap_lut = remap_lut

        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 0  # additional frames are fused with target-frame. Hence, multiscan+1 point clouds in total
        print('multiscan: %d' %multiscan)
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()
        for i_folder in split:
            # velodyne path corresponding to voxel path
            complete_path = os.path.join(data_path, str(i_folder).zfill(2), "voxels")
            files = list(pathlib.Path(complete_path).glob('*.bin'))
            for filename in files:
                self.im_idx.append(str(filename).replace('voxels', 'velodyne'))



    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))  # point cloud
        origin_len = len(raw_data)
        voxel_label = 1

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        for fuse_idx in range(self.multiscan):
            plus_idx = fuse_idx + 1
            data_idx = number_idx + plus_idx
            path = self.im_idx[index][:-10]
            newpath2 = path + str(data_idx).zfill(6) + self.im_idx[index][-4:]
            voxel_path = path.replace('velodyne', 'labels')
            files = list(pathlib.Path(voxel_path).glob('*.label'))
            
            if data_idx < len(files):
                pose = self.poses[dir_idx][data_idx]

                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if voxel_label == 0:
                    if self.imageset == 'test':
                        annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                    else:
                        annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                      dtype=np.int32).reshape((-1, 1))
                        annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    if voxel_label == 0:
                        annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        if self.imageset == 'test':
            annotated_data = np.zeros([256, 256, 32], dtype=int).reshape((-1, 1))
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'voxels')[:-3] + 'label',
                                                 dtype=np.uint16).reshape((-1, 1))  # voxel labels
                                                 
        annotated_data = self.comletion_remap_lut[annotated_data]
        annotated_data = annotated_data.reshape((256, 256, 32))

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))  # xyz, voxel labels

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len)  # origin_len is used to indicate the length of target-scan

        return data_tuple
import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from mmcv.image.io import imread

@register_dataset
class nusc(data.Dataset):
    def __init__(self, data_path, imageset='train', label_mapping="nuscenes.yaml", nusc=None):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'],
        }
        
        # read 6 cams
        # imgs = []
        # for filename in imgs_info['img_filename']:
        #     imgs.append(
        #         imread(filename, 'unchanged').astype(np.float32)
        #     )
        
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.data_path, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        # points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        # points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        #add Workshop label
        path = os.path.join(self.data_path,imgs_info['gt'])
        points_label = np.load(path)
        semantics = points_label['semantics']
        mask_lidar = points_label['mask_lidar']
        mask_camera = points_label['mask_camera']
        
        lidar_path = info['lidar_path']        
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
        lidar2ego = imgs_info['lidar2ego']
        points_img = np.matmul(points[:, :3], lidar2ego[:3, :3].T) + lidar2ego[:3, 3][None]
        data_tuple = (points_img, semantics.astype(np.uint8), points[:, 3], info) # xyz, voxel labels
        #print(f"ITEMS FROM NUSC: {len(data_tuple)}")
        
        return data_tuple
    
    def get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
        """
        # standard protocal modified from SECOND.Pytorch
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(lidar2ego_rotation).rotation_matrix
        lidar2lidarego[:3, 3] = lidar2ego_translation
        # lidar2ego = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation))
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            gt=info['occ_gt_path'],
            lidar2ego = lidar2lidarego,
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict

# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
