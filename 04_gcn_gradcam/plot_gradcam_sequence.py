import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from pyskl.apis.inference import init_recognizer
from pyskl.datasets.pipelines import Compose
import pickle

from gradcam import GradCAM

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def build_inputs(model, raw_data):

    cfg = model.cfg
    device = next(model.parameters()).device

    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    original_pipeline = []
    original_pipeline.append({'type': 'GenSkeFeat', 'dataset': 'coco', 'feats': ['j']})
    original_pipeline.append({'type': 'UniformSample', 'clip_len': 80, 'num_clips':10, 'test_mode': True})
    original_pipeline.append({'type': 'PoseDecode'})
    original_pipeline.append({'type': 'FormatGCNInput', 'num_person': 2})
    original_pipeline = Compose(original_pipeline)

    start_index = cfg.data.test.get('start_index', 0)
    data = dict(
        keypoint = raw_data['keypoint'],
        keypoint_score = raw_data['keypoint_score'],
        img_shape = raw_data['img_shape'],
        total_frames = raw_data['total_frames'],
        start_index = start_index, 
        label = raw_data['label']
    )

    original_data = dict(
        keypoint = raw_data['keypoint'],
        keypoint_score = raw_data['keypoint_score'],
        img_shape = raw_data['img_shape'],
        total_frames = raw_data['total_frames'],
        start_index = start_index, 
        label = raw_data['label']
    )

    original_data = original_pipeline(original_data)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    original_data['total_frames'] = 80

    return data, original_data

def main():
    device = 'cuda:0'

    target_true_label = 2
    age_idx='B'
    tp_label_idx = f'{age_idx}_tp_label_{target_true_label}'
    
    action_idx = '1'

    cfg_path = f"work_dirs/my_skeleton/{age_idx}/{action_idx}/fold_16/fold_16.py"
    cfg = Config.fromfile(cfg_path)
    checkpoint = f'work_dirs/my_skeleton/{age_idx}/{action_idx}/fold_16/latest.pth'
    keypoint_path = f'01_skeleton_extraction/my_skeleton/{age_idx}/{action_idx}/fold_16_test.pkl'
    target_layer_name = 'backbone.net.9'

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    model = init_recognizer(cfg, checkpoint, device=device)
    if not (osp.exists(keypoint_path) or keypoint_path.startswith('http')):
        raise RuntimeError(f"'{keypoint_path} is missing")

    with open(keypoint_path,"rb") as fr:
        data = pickle.load(fr)

    total_data_len = 0
    cnt_arr = np.zeros(17)
    
    agg_data = data
    for i in range(len(agg_data)):
        video_name = data[i]['frame_dir']
        video_true_label = agg_data[i]['label']

        if video_true_label != target_true_label:
            continue

        inputs, original = build_inputs(model, agg_data[i])
        video_name = agg_data[i]['frame_dir']
        gradcam = GradCAM(model, target_layer_name)

        # the code below takes long time, so make condition for the target videos is desirable.
        localization_results, imgs_results, heatmap, isLabelSame, raw_frame = gradcam(inputs, original, use_labels=True)

        if not isLabelSame:
            continue
        
        # make each plot of the frame
        os.makedirs(f"05_gcn_gradcam/gradcam_sequence/{tp_label_idx}/{action_idx}/{video_name}", exist_ok=True)
        for i in range(len(raw_frame)):
            plt.imsave(f"05_gcn_gradcam/gradcam_sequence/{tp_label_idx}/{action_idx}/{video_name}/{i}.svg", raw_frame[i], dpi=300,format='svg')

        counts = pd.DataFrame(columns=['head', 'right arm', 'left arm', 'right leg', 'left leg', 'label', 'action'])
        localization_results = localization_results[:, 0, :, :].squeeze().cpu().numpy()

        for j in range(10):
            for k in range(80):
                mx = -1000
                mx_idx = 0
                for l in range(17):
                    if mx <= localization_results[j][k][l]:
                        mx_idx = l
                        mx = localization_results[j][k][l]
                cnt_arr[mx_idx] += 1
        

        heatmap = heatmap[0, :, :, :].squeeze()
        heatmap = heatmap.transpose(1, 0, 2)

        fig, ax = plt.subplots(figsize=(22,6))
        im = ax.imshow(heatmap, cmap='magma')
        cbaxes = fig.add_axes([0.93, .15, .015, .7])
        cbar = ax.figure.colorbar(im, ax=ax, cax = cbaxes)
        ax.set_xlabel("Frames", fontsize=28)
        ax.set_ylabel("Joint Indices", fontsize=28)
        ax.set_yticks(range(0,17))
        dim = np.arange(1, 17, 1)
        ax.set_xticks(range(0,80,1))

        ax.set_facecolor('gray')

        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.tick_params(axis='y', which='major', pad=5)
        ax.grid(which = 'minor', color = 'w')
        os.makedirs(f"05_gcn_gradcam/gradcam_sequence/{tp_label_idx}/{action_idx}", exist_ok=True)
        fig.savefig(f"05_gcn_gradcam/gradcam_sequence/{tp_label_idx}/{action_idx}/{video_name}_heatmap.svg", dpi=300, transparent=True, format='svg',bbox_inches='tight', pad_inches=0)
    
    cnt_pd = pd.DataFrame(cnt_arr)
    cnt_pd.to_csv(f"05_gcn_gradcam/gradcam_sequence/{tp_label_idx}/cnt_{action_idx}.csv", index=False)

if __name__ == '__main__':
    main()