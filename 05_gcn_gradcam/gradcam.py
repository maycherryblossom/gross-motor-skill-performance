import torch
import torch.nn as nn
import torch.nn.functional as F
from pyskl.utils.visualize import Vis3DPose, Vis2DPose, Vis2DPoseWHeatmap
import numpy as np

class GradCAM:
    '''
    gradcam for recognizer gcn.
    '''
    def __init__(self, model, target_layer_name, colormap='magma'):
        from pyskl.models.recognizers import RecognizerGCN
        if isinstance(model, RecognizerGCN):
            self.is_recognizergcn = True
        else:
            raise ValueError('This is for GCN ONLY!!!!')

        self.model = model
        self.model.eval()
        self.target_gradients = None
        self.target_activations = None


        import matplotlib.pyplot as plt
        self.colormap = plt.get_cmap(colormap)  
        self._register_hooks(target_layer_name)
    

    def _register_hooks(self, layer_name):

        def get_gradients(module, grad_input, grad_output):
            self.target_gradients = grad_output[0].detach()

        def get_activations(module, input, output):
            self.target_activations = output.clone().detach()

        layer_ls = layer_name.split('.')
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]
        
        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _calculate_localizeation_map(self, inputs, use_labels, delta=1e-20):

        inputs['keypoint'] = inputs['keypoint'].clone()

        self.model.test_cfg['average_clips'] = 'score'
        preds = self.model(gradcam=True, **inputs)
        
        isLabelSame = False
        if (use_labels):
            labels = inputs['label']
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            if torch.max(preds, dim=-1)[0] == torch.gather(preds, dim=1, index=labels):
                isLabelSame = True
            score = torch.gather(preds, dim=1, index=labels)
        else:
            score = torch.max(preds, dim=-1)[0]

        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()

        bs, N, M, T, V, C = inputs['keypoint'].size()
        gradients = self.target_gradients
        activations = self.target_activations

        NM, C, tg, V = gradients.size()

        gradients = gradients.permute(0, 2, 1, 3)
        activations = activations.permute(0, 2, 1, 3)

        weights = torch.mean(gradients.view(N*M, tg, C, -1), dim=3)
        weights = weights.view(N*M, tg, C, 1)

        activations = activations.view([N*M, tg, C] + list(activations.size()[-1:]))

        localization_map = torch.sum(weights * activations, dim=2, keepdim=True)

        localization_map = F.relu(localization_map)
        localization_map = localization_map.permute(0, 2, 1, 3)
        localization_map = F.interpolate(
            localization_map,
            size=(T, V),
            mode='bilinear',
            align_corners=False)

        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(N*M, -1), dim=-1, keepdim=True)[0],
            torch.max(localization_map.view(N*M, -1), dim=-1, keepdim=True)[0])
        localization_map_min = torch.reshape(
            localization_map_min, shape=(N*M, 1, 1, 1))
        localization_map_max = torch.reshape(
            localization_map_max, shape=(N*M, 1, 1, 1))
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + delta)
        localization_map = localization_map.data
        localization_map = localization_map.view(N, M, 1, T, V)
        return localization_map.squeeze(dim=2), preds, isLabelSame

    def _alpha_blending(self, localization_map, input_for_vids, alpha):

        def color_keypoints(localization_map, input_keypoints):
            heatmap = self.colormap(localization_map.detach().numpy())
            heatmap = heatmap[:, :, :, :3]
            heatmap = torch.from_numpy(heatmap)
            final_coordinates = []

            for m in range(input_keypoints.shape[0]):
                for t in range(input_keypoints.shape[1]):
                    for v in range(input_keypoints.shape[2]):
                        coordinate = input_keypoints[m, t, v]
                        color = heatmap[m, t, v]
                        final_coordinates.append(np.hstack((coordinate, color)))
            return final_coordinates

        localization_map = localization_map.cpu()
        input_keypoints= input_for_vids['keypoint']

        final_coordinates = color_keypoints(localization_map, input_keypoints) 

        heatmap = self.colormap(localization_map.detach().numpy())
        heatmap = heatmap[:, :, :, :3] 
        vid, raw_frames = Vis2DPoseWHeatmap(input_for_vids, thre=0.2, out_shape=(540, 960), layout='coco', fps=24, video=None, heatmap=heatmap)

        return vid, heatmap, raw_frames

    def __call__(self, inputs, original, use_labels=False, alpha=0.5):

        localization_map, preds, isLabelSame = self._calculate_localizeation_map(inputs, use_labels=use_labels)
        inputs_for_vid = original.copy()
        inputs_for_vid['keypoint'] = inputs_for_vid['keypoint'][0, :, :, :, :].squeeze()
        localization_map_for_vid = localization_map[0, :, :, :]

        blended_imgs, heatmap, raw_frames = self._alpha_blending(localization_map_for_vid, inputs_for_vid,
                                            alpha)

        return localization_map, blended_imgs, heatmap, isLabelSame, raw_frames
        
