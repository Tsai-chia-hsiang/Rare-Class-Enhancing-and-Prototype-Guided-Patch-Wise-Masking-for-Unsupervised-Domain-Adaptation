# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn
import mmcv
import os.path as osp
import json

def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    original_freq = freq
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy(), original_freq

def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def get_rare_class_mask(labels, class_probabilities,rcs_classes):
    class_masks = []
    for label in labels:
        sorted_prob = []
        classes = torch.unique(labels)
        # mmcv.print_log(f'classes: {classes}', 'mmseg')
        # mmcv.print_log(f'classes.shape: {classes.shape}', 'mmseg')
        # mmcv.print_log(f'rcs_classes111: {rcs_classes}', 'mmseg')
        # mmcv.print_log(f'classes111: {classes}', 'mmseg')
        # nclasses = classes.shape[0]
       
        sorted_classes = torch.tensor([x for x in rcs_classes if x in classes.tolist()])
        sorted_classes = sorted_classes.to(label.device)
        
        # mmcv.print_log(f'sorted_classes: {sorted_classes}', 'mmseg')
        
        
        sorted_classes_list = sorted_classes.tolist()
        rcs_mapping = dict(zip(rcs_classes, class_probabilities))
        # mmcv.print_log(f'rcs_mapping: {rcs_mapping}', 'mmseg')
        probs = [rcs_mapping.get(cls) for cls in sorted_classes_list]
        # mmcv.print_log(f'probs: {probs}', 'mmseg')
        # 歸一化概率值
        sorted_prob = probs / np.sum(probs)
      
        # mmcv.print_log(f'rcs_classes: {rcs_classes}', 'mmseg')
        # mmcv.print_log(f'nclasses: {nclasses}', 'mmseg')
        # mmcv.print_log(f'sorted_prob: {sorted_prob}', 'mmseg')
        # mmcv.print_log(f'sorted_prob: {sorted_prob}', 'mmseg')
        # mmcv.print_log(f'sorted_classes: {sorted_classes}', 'mmseg')
        # mmcv.print_log(f'sorted_classes.shape: {sorted_classes.shape}', 'mmseg')
        nclasses = sorted_classes.shape[0]
        class_choice = np.random.choice(nclasses, p=sorted_prob, size=int((nclasses + nclasses % 2) / 2), replace=False)
        classes = sorted_classes[torch.Tensor(class_choice).long()]
        
        # mmcv.print_log(f'class_choice: {class_choice}', 'mmseg')
        # mmcv.print_log(f'sorted_classes: {sorted_classes}', 'mmseg')
        # mmcv.print_log(f'classes: {classes}', 'mmseg')
        
        # mmcv.print_log(f'class_choice: {class_choice}', 'mmseg')
        # mmcv.print_log(f'label: {label}', 'mmseg')
        
        classes = classes.to(label.device)

        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        # mmcv.print_log(f'class_masks: {class_masks}', 'mmseg') 
    return class_masks  

def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
