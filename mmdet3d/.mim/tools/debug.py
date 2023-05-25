from os import path as osp

import mmcv
import numpy as np
import pickle

if __name__ == '__main__':
    data1 = mmcv.load('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', file_format='pkl')
    # data_infos1 = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # print()
    # breakpoint()
    data2 = mmcv.load('data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', file_format='pkl')

    data = data1['infos'] + data2['infos']
    data_infos = list(sorted(data, key=lambda e: e['timestamp']))
    print(len(data_infos))

    save_dict = {
        'infos': data_infos,
        'metadata': data1['metadata']
    }

    with open('./data/nuscenes/bevdetv2-nuscenes_infos_trainval.pkl', 'wb') as fid:
        pickle.dump(save_dict, fid)