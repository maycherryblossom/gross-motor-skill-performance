import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test, single_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port

import pickle
import numpy as np
import pandas as pd
import random

def make_dataset(cfg):
    # Load eval_config from cfg
    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    dataset_test = build_dataset(cfg.data.test, dict(test_mode=True))
    dataset_train = build_dataset(cfg.data.train, dict(test_mode=True))
    dataset_val = build_dataset(cfg.data.val, dict(test_mode=True))

    with open(cfg.data.test.ann_file,"rb") as fr:
        data_orig_test = pickle.load(fr)

    with open(cfg.data.test.ann_file[:-9]+'_'+'train'+'.pkl',"rb") as fr:
        data_orig_train = pickle.load(fr)
    
    with open(cfg.data.test.ann_file[:-9]+'_'+'val'+'.pkl',"rb") as fr:
        data_orig_val = pickle.load(fr)

        
    return dataset_test, dataset_train, dataset_val, data_orig_test, data_orig_train, data_orig_val

def set_dataloader(cfg, dataset_test, dataset_train, dataset_val):
    dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    shuffle=False)
    dataloader_setting_test = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    dataloader_setting_train = dict(dataloader_setting, **cfg.data.get('train_dataloader', {}))
    dataloader_setting_val = dict(dataloader_setting, **cfg.data.get('val_dataloader', {}))

    data_loader_test = build_dataloader(dataset_test, **dataloader_setting_test)
    data_loader_train = build_dataloader(dataset_train, **dataloader_setting_train)
    data_loader_val = build_dataloader(dataset_val, **dataloader_setting_val)

    

    return data_loader_test, data_loader_train, data_loader_val

def load_model_and_predict(cfg, checkpoint, data_loader_test, data_loader_train, data_loader_val):
    features = {}

    def get_features(name): 
        def hook(model, input, output):
            features[name] = output
        return hook

    model = build_model(cfg.model).cuda()
    _ = load_checkpoint(model, checkpoint, map_location={'cuda:0': 'cpu'})
    # model = build_model(cfg.model).cuda()
    # load_checkpoint(model.cuda(), checkpoint, map_location='cuda:0')
    
    # model.backbone.net[8].tcn1.register_forward_hook(get_features('tcn1'))
    model.eval()

    PREDS_train = []
    FEATS_train = []
    results_train = []
    dataset_train = data_loader_train.dataset
    prog_bar = mmcv.ProgressBar(len(dataset_train))
    for data in data_loader_train:
        data = data['keypoint'].to('cuda:0')
        with torch.no_grad():
            # outputs = single_gpu_test(model.cpu(), data_loader)
            result = model(return_loss=False, keypoint = data)
            PREDS_train.append(result)
            # print(result)
            # FEATS_train.append(features['tcn1'])
        results_train.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()


    PREDS_test = []
    FEATS_test = []
    results_test = []
    dataset_test = data_loader_test.dataset
    prog_bar = mmcv.ProgressBar(len(dataset_test))
    for data in data_loader_test:
        data = data['keypoint'].to('cuda:0')
        with torch.no_grad():
            # outputs = single_gpu_test(model.cpu(), data_loader)
            result = model(return_loss=False, keypoint=data)
            PREDS_test.append(result)
            # print(result)
            # FEATS_test.append(features['tcn1'])
        results_test.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    PREDS_val = []
    FEATS_val = []
    results_val = []
    dataset_val = data_loader_val.dataset
    prog_bar = mmcv.ProgressBar(len(dataset_val))
    for data in data_loader_val:
        data = data['keypoint'].to('cuda:0')
        with torch.no_grad():
            # outputs = single_gpu_test(model.cpu(), data_loader)
            result = model(return_loss=False, keypoint=data)
            PREDS_val.append(result)
            # print(result)
            # FEATS_val.append(features['tcn1'])
        results_val.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    
    return results_test, results_train, results_val

def make_dataframe(action_num, results_test, results_train, results_val, data_orig_test, data_orig_train, data_orig_val):
    col1 = f'{action_num}_1'
    col2 = f'{action_num}_2'
    col3 = f'{action_num}_3'

    agg_test = pd.DataFrame(columns=['pid', col1, col2, col3])
    agg_train = pd.DataFrame(columns=['pid', col1, col2, col3])
    agg_val = pd.DataFrame(columns=['pid', col1, col2, col3])


    out_test = np.array(results_test)
    cnt = 0
    output = out_test
    originalData = data_orig_test
    for i in range(len(out_test)):
        pid = originalData[i]['frame_dir'][:4]
        aid = originalData[i]['frame_dir'][-5:-4]
        agg_test.loc[cnt] = [pid, output[i][0], output[i][1], output[i][2]]
        cnt += 1

        
    out_train = np.array(results_train)
    out_train = out_train[:int((out_train.shape[0])/5), :]
    cnt = 0
    output = out_train
    originalData = data_orig_train
    for i in range(len(out_train)):
        pid = originalData[i]['frame_dir'][:4]
        aid = originalData[i]['frame_dir'][-5:-4]
        agg_train.loc[cnt] = [pid, output[i][0], output[i][1], output[i][2]]
        cnt += 1

    out_val = np.array(results_val)
    cnt = 0
    output = out_val
    originalData = data_orig_val
    for i in range(len(out_val)):
        pid = originalData[i]['frame_dir'][:4]
        aid = originalData[i]['frame_dir'][-5:-4]
        agg_val.loc[cnt] = [pid, output[i][0], output[i][1], output[i][2]]
        cnt += 1

    return agg_test, agg_train, agg_val


def CustomMerge(df1, df2, df3, df4):
    df_concat = pd.concat([df1, df2, df3, df4], sort=False)
    pidLi = np.array(df_concat['pid'].sort_values().unique())
    newDf = pd.DataFrame(columns=['pid', '1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3'])
    prev_arr = newDf
    for item in pidLi:
        # print(df1[df1['pid'] == item])
        # print(df2[df2['pid'] == item])
        arr1 = df1[df1['pid'] == item].reset_index().drop(columns=['index'])
        arr2 = df2[df2['pid'] == item].drop(columns=['pid']).reset_index().drop(columns=['index'])
        arr3 = df3[df3['pid'] == item].drop(columns=['pid']).reset_index().drop(columns=['index'])
        arr4 = df4[df4['pid'] == item].drop(columns=['pid']).reset_index().drop(columns=['index'])
        # print(pd.concat([arr1, arr2, arr3, arr4], axis=1))
        
        # cat_arr = pd.concat([arr1, arr2, arr3, arr4], axis=1).apply(lambda x: x.fillna(random.sample(list(x.dropna()), 1)[0]))
        cat_arr = pd.concat([arr1, arr2, arr3, arr4], axis=1)
        cat_arr.dropna(how='any', inplace=True)

        # cat_arr = cat_arr.apply(lambda x: x.fillna(random.sample(list(x.dropna()), 1)[0]) if len(list(x.dropna()))>0 else x.fillna(prev_arr[x.name].mean()))
        arr1_sz = len(arr1.dropna())
        arr2_sz = len(arr2.dropna())
        arr3_sz = len(arr3.dropna())
        arr4_sz = len(arr4.dropna())
        min_len = min(arr1_sz, arr2_sz, arr3_sz, arr4_sz)
        max_len = max(arr1_sz, arr2_sz, arr3_sz, arr4_sz)

        last_two_rows_index = cat_arr.index[-2:]
        if ((max_len - min_len) > 3) :
            cat_arr.drop(last_two_rows_index)

        cat_arr = cat_arr.apply(lambda x: x.ffill() if x.name == 'pid' else (x.fillna(random.sample(list(x.dropna()), 1)[0]) if len(list(x.dropna()))>0 else x))
        # print(cat_arr)
        new_arr = pd.concat([prev_arr, cat_arr], axis=0)
        prev_arr = new_arr
        # print(new_arr)
        
    return prev_arr

def label_and_save(age, shortMerge_test, shortMerge_train, shortMerge_val, i):
    # ev = pd.read_csv('대근육총합및4수준_processed.csv')
    # ev = ev[['아동영상코드', '대근육 결과4수준']]
    # ev = ev.dropna(axis=0)
    # ev = ev[ev.아동영상코드.str[0] == 'B']s

    newlabel = pd.read_csv("02_labels/development_status_label.csv")
    newlabel = newlabel[['ID', 'label_pid']]
    newlabel = newlabel.drop_duplicates()

    short_train_labeled = pd.merge(shortMerge_train, newlabel, how="inner", left_on="pid", right_on="ID")
    short_test_labeled = pd.merge(shortMerge_test, newlabel, how="inner", left_on="pid", right_on="ID")
    short_val_labeled = pd.merge(shortMerge_val, newlabel, how="inner", left_on="pid", right_on="ID")

    short_train_labeled.drop(columns=['ID'], inplace=True)
    # short_train_labeled.drop(columns=['pid'], inplace=True)
    short_test_labeled.drop(columns=['ID'], inplace=True)
    # short_test_labeled.drop(columns=['pid'], inplace=True)
    short_val_labeled.drop(columns=['ID'], inplace=True)
    # short_val_labeled.drop(columns=['pid'], inplace=True)

    return short_train_labeled, short_test_labeled, short_val_labeled
    
def main ():
    random.seed(0)

    age = 'B'
    folder_specific = 'my_skeleton'
    path = f'work_dirs/{folder_specific}/{age}/'
    for i in range (1, 29):

        action1_path = path + '1/' + f'fold_{i}/'
        action2_path = path + '2/' + f'fold_{i}/'
        action3_path = path + '3/' + f'fold_{i}/'
        action4_path = path + '4/' + f'fold_{i}/'

        cfg1 = Config.fromfile(action1_path + f'fold_{i}.py')
        cfg2 = Config.fromfile(action2_path + f'fold_{i}.py')
        cfg3 = Config.fromfile(action3_path + f'fold_{i}.py')
        cfg4 = Config.fromfile(action4_path + f'fold_{i}.py')

        for filename in os.listdir(action1_path):
            if filename.startswith("best_top1"):
                checkpoint1 = cache_checkpoint(action1_path + filename)
        for filename in os.listdir(action2_path):
            if filename.startswith("best_top1"):
                checkpoint2 = cache_checkpoint(action2_path + filename)
        for filename in os.listdir(action3_path):
            if filename.startswith("best_top1"):
                checkpoint3 = cache_checkpoint(action3_path + filename)
        for filename in os.listdir(action4_path):
            if filename.startswith("best_top1"):
                checkpoint4 = cache_checkpoint(action4_path + filename)

        # checkpoint1 = cache_checkpoint(action1_path + 'latest.pth')
        # checkpoint2 = cache_checkpoint(action2_path + 'latest.pth')
        # checkpoint3 = cache_checkpoint(action3_path + 'latest.pth')
        # checkpoint4 = cache_checkpoint(action4_path + 'latest.pth')

        # B 행동 1
        dataset_test_act1, dataset_train_act1, dataset_val_act1, data_orig_test_act1, data_orig_train_act1, data_orig_val_act1 = make_dataset(cfg1)
        data_loader_test_act1, data_loader_train_act1, data_loader_val_act1 = set_dataloader(cfg1, dataset_test_act1, dataset_train_act1, dataset_val_act1)
        results_test_act1, results_train_act1, results_val_act1 = load_model_and_predict(cfg1, checkpoint1, data_loader_test_act1, data_loader_train_act1, data_loader_val_act1)
        agg_test_act1, agg_train_act1, agg_val_act1 = make_dataframe(1, results_test_act1, results_train_act1, results_val_act1, data_orig_test_act1, data_orig_train_act1, data_orig_val_act1)

        # B 행동 2
        dataset_test_act2, dataset_train_act2, dataset_val_act2, data_orig_test_act2, data_orig_train_act2, data_orig_val_act2 = make_dataset(cfg2)
        data_loader_test_act2, data_loader_train_act2, data_loader_val_act2 = set_dataloader(cfg2, dataset_test_act2, dataset_train_act2, dataset_val_act2)
        results_test_act2, results_train_act2, results_val_act2 = load_model_and_predict(cfg2, checkpoint2, data_loader_test_act2, data_loader_train_act2, data_loader_val_act2)
        agg_test_act2, agg_train_act2, agg_val_act2 = make_dataframe(2, results_test_act2, results_train_act2, results_val_act2, data_orig_test_act2, data_orig_train_act2, data_orig_val_act2)

        # B 행동 3
        dataset_test_act3, dataset_train_act3, dataset_val_act3, data_orig_test_act3, data_orig_train_act3, data_orig_val_act3 = make_dataset(cfg3)
        data_loader_test_act3, data_loader_train_act3, data_loader_val_act3 = set_dataloader(cfg3, dataset_test_act3, dataset_train_act3, dataset_val_act3)
        results_test_act3, results_train_act3, results_val_act3 = load_model_and_predict(cfg3, checkpoint3, data_loader_test_act3, data_loader_train_act3, data_loader_val_act3)
        agg_test_act3, agg_train_act3, agg_val_act3 = make_dataframe(3, results_test_act3, results_train_act3, results_val_act3, data_orig_test_act3, data_orig_train_act3, data_orig_val_act3)

        # B 행동 4
        dataset_test_act4, dataset_train_act4, dataset_val_act4, data_orig_test_act4, data_orig_train_act4, data_orig_val_act4 = make_dataset(cfg4)
        data_loader_test_act4, data_loader_train_act4, data_loader_val_act4 = set_dataloader(cfg4, dataset_test_act4, dataset_train_act4, dataset_val_act4)
        results_test_act4, results_train_act4, results_val_act4 = load_model_and_predict(cfg4, checkpoint4, data_loader_test_act4, data_loader_train_act4, data_loader_val_act4)
        agg_test_act4, agg_train_act4, agg_val_act4 = make_dataframe(4, results_test_act4, results_train_act4, results_val_act4, data_orig_test_act4, data_orig_train_act4, data_orig_val_act4)

        
        fold_path = f"04_performance_and_gradcam_gcn/{folder_specific}/{age}/fold_{i}"

        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        agg_train_act1.to_csv(f'{fold_path}/agg_train_act1.csv', index=False, encoding="utf-8-sig")
        agg_test_act1.to_csv(f'{fold_path}/agg_test_act1.csv', index=False, encoding="utf-8-sig")
        agg_val_act1.to_csv(f'{fold_path}/agg_val_act1.csv', index=False, encoding="utf-8-sig")

        agg_train_act2.to_csv(f'{fold_path}/agg_train_act2.csv', index=False, encoding="utf-8-sig")
        agg_test_act2.to_csv(f'{fold_path}/agg_test_act2.csv', index=False, encoding="utf-8-sig")
        agg_val_act2.to_csv(f'{fold_path}/agg_val_act2.csv', index=False, encoding="utf-8-sig")

        agg_train_act3.to_csv(f'{fold_path}/agg_train_act3.csv', index=False, encoding="utf-8-sig")
        agg_test_act3.to_csv(f'{fold_path}/agg_test_act3.csv', index=False, encoding="utf-8-sig")
        agg_val_act3.to_csv(f'{fold_path}/agg_val_act3.csv', index=False, encoding="utf-8-sig")

        agg_train_act4.to_csv(f'{fold_path}/agg_train_act4.csv', index=False, encoding="utf-8-sig")
        agg_test_act4.to_csv(f'{fold_path}/agg_test_act4.csv', index=False, encoding="utf-8-sig")
        agg_val_act4.to_csv(f'{fold_path}/agg_val_act4.csv', index=False, encoding="utf-8-sig")

        shortMerge_test = CustomMerge(agg_test_act1, agg_test_act2, agg_test_act3, agg_test_act4)
        shortMerge_train = CustomMerge(agg_train_act1, agg_train_act2, agg_train_act3, agg_train_act4)
        shortMerge_val = CustomMerge(agg_val_act1, agg_val_act2, agg_val_act3, agg_val_act4)

        short_train_labeled, short_test_labeled, short_val_labeled = label_and_save("B", shortMerge_test, shortMerge_train, shortMerge_val, i)

        short_train_labeled.to_csv(f'{fold_path}/short_train_labeled.csv', index=False, encoding="utf-8-sig")
        short_test_labeled.to_csv(f'{fold_path}/short_test_labeled.csv', index=False, encoding="utf-8-sig")
        short_val_labeled.to_csv(f'{fold_path}/short_val_labeled.csv', index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    
    main()