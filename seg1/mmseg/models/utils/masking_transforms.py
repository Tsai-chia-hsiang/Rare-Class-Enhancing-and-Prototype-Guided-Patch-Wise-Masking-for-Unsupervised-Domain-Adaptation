# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import mmcv
from mmseg.ops import resize
import numpy as np
import torch.nn.functional as F

def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size, usedCL,
        r_0 ,
        r_final,
        total_iteration):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.usedCL = usedCL
        self.r_0 = r_0
        self.r_final = r_final
        self.total_iteration = total_iteration
        
    @torch.no_grad()    
    def CL(self,local_iter):
        # mmcv.print_log(f'local_iter: {local_iter}', 'mmseg')
        # mmcv.print_log(f'usedCL: {self.usedCL}', 'mmseg')
        # mmcv.print_log(f'r_0: {self.r_0}', 'mmseg')
        # mmcv.print_log(f'r_final: {self.r_final}', 'mmseg')
        # mmcv.print_log(f'total_iteration: {self.total_iteration }', 'mmseg')       
        self.mask_ratio = self.r_0 + (self.r_final - self.r_0)*(local_iter/(self.total_iteration-1))           

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape
        
        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)

        input_mask = (input_mask > self.mask_ratio).float()   
        # mmcv.print_log(f'self.mask_ratio_rand_patch: {self.mask_ratio }', 'mmseg') 
        input_mask = resize(input_mask, size=(H, W))
        return input_mask
    
    @torch.no_grad()
    def generate_original_pixelwise_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, H, W
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        # mmcv.print_log(f'originalpixelwise: {self.mask_ratio}', 'mmseg')
        # mmcv.print_log(f'input_mask: {input_mask}', 'mmseg')

        # array = input_mask.cpu().clone().numpy()
        # zero_ratio = np.mean(array == 0)
        # one_ratio = np.mean(array == 1)

       
        # mmcv.print_log(f'zero_ratio: {zero_ratio}', 'mmseg')
        # mmcv.print_log(f'one_ratio: {one_ratio}', 'mmseg')
        return input_mask
    
    @torch.no_grad()
    def generate_proto_mask(self, imgs, Proto, feat, local_iter):
        B, _, H, W = imgs.shape
        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)
        # mmcv.print_log(f'feat.size(): {feat.size()}', 'mmseg')
        # mmcv.print_log(f'Proto.size(): {Proto.size()}', 'mmseg')
        logits = feat.mm(Proto.permute(1, 0).contiguous())
        del feat, Proto
        logits = torch.softmax(logits, dim=1)

        confidence, _ = torch.max(logits, dim=1)
        del logits
        # mmcv.print_log(f'confidence.size(): {confidence.size()}', 'mmseg')
        # mmcv.print_log(f'confidence: {confidence}', 'mmseg')
        max = torch.max(confidence, dim=0)
        # mmcv.print_log(f'max: {max}', 'mmseg')

        confidence = confidence.reshape(B, round(H / self.mask_block_size)*round(
            W / self.mask_block_size))
        
        # mmcv.print_log(f'confidence.size(): {confidence.size()}', 'mmseg')
        # mmcv.print_log(f'confidence: {confidence}', 'mmseg')
        N = int(confidence.shape[1]* self.mask_ratio)
        idx = torch.argsort(confidence, descending=True)[:,:N]
        # mmcv.print_log(f'idx: {idx, type(idx)}', 'mmseg')
        proto_mask = torch.ones(confidence.shape, device=confidence.device)
        del confidence, N 
        proto_mask.scatter_(1, idx, False)
        # mmcv.print_log(f'proto_mask: {proto_mask}', 'mmseg')
        # mmcv.print_log(f'proto_mask.size: {proto_mask.size()}', 'mmseg')
        del idx
        proto_mask = proto_mask.reshape(B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)) 
        proto_mask = proto_mask.float()
        proto_mask = resize(proto_mask, size=(H, W))
       
      
        return proto_mask
    
    
    @torch.no_grad()
    def generate_proto_prob_mask(self, imgs, Proto, feat, local_iter):
        B, _, H, W = imgs.shape
        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)
        # mmcv.print_log(f'feat.size(): {feat.size()}', 'mmseg')
        # mmcv.print_log(f'Proto.size(): {Proto.size()}', 'mmseg')
        logits = feat.mm(Proto.permute(1, 0).contiguous())
        del feat, Proto
        logits = torch.softmax(logits, dim=1)

        confidence, _ = torch.max(logits, dim=1)
        del logits
        
        
        # mmcv.print_log(f'confidence1: {confidence}', 'mmseg')
        confidence = confidence.reshape(B, round(H / self.mask_block_size)*round(
            W / self.mask_block_size))
        confidence = torch.softmax(confidence/ 0.1, dim=1)
        # mmcv.print_log(f'confidence2.size(): {confidence.size()}', 'mmseg')
        
        # mmcv.print_log(f'confidence2: {confidence}', 'mmseg')

        # max = torch.max(confidence, dim=0)
        # mmcv.print_log(f'max: {max}', 'mmseg')
        N = int(round(H / self.mask_block_size)*round(
            W / self.mask_block_size)* self.mask_ratio)
        # idx = torch.argsort(confidence, descending=True)[:,:N]
        
        selected_indices=[]
        for i in range(B):
            confidence_flat = np.ravel(confidence[i].cpu())
            confidence_flat = confidence_flat / confidence_flat.sum()
            # if i==0:
            #     mmcv.print_log(f'confidence_flat0.size: {len(confidence_flat)}', 'mmseg')
            #     mmcv.print_log(f'confidence_flat0: {confidence_flat}', 'mmseg')
            # else:
            #     mmcv.print_log(f'confidence_flat1.size: {len(confidence_flat)}', 'mmseg')
            #     mmcv.print_log(f'confidence_flat1: {confidence_flat}', 'mmseg')    
            
            selected_indices_batch = np.random.choice(confidence_flat.size, size=N, replace=False, p=confidence_flat)
            selected_indices.append(selected_indices_batch)
        del confidence_flat, selected_indices_batch     
        
        selected_indices = np.array(selected_indices)
        selected_indices = torch.from_numpy(selected_indices)
        
        # mmcv.print_log(f'selected_indices: {selected_indices.shape}', 'mmseg')       

        proto_mask = torch.ones(confidence.shape, device=confidence.device)
        del confidence, N 
        
        selected_indices=selected_indices.to(proto_mask.device)
        proto_mask.scatter_(1, selected_indices, False)
        del selected_indices
        # mmcv.print_log(f'proto_mask: {proto_mask}', 'mmseg')
        # mmcv.print_log(f'proto_mask.size: {proto_mask.size()}', 'mmseg')
        
        proto_mask = proto_mask.reshape(B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)) 
        proto_mask = proto_mask.float()
        # mmcv.print_log(f'proto_mask2.size: {proto_mask.size()}', 'mmseg')
        proto_mask = resize(proto_mask, size=(H, W))

        return proto_mask
    
    
    
    
    @torch.no_grad()
    def mask_image(self, imgs, mask_type, proto, target_rep, local_iter):
        # mmcv.print_log(f'mask_type: {mask_type}', 'mmseg')
        if self.usedCL:
            self.CL(local_iter)
        if mask_type == 'original':
            input_mask = self.generate_mask(imgs)
        elif mask_type == 'original_pixelwise':
            input_mask = self.generate_original_pixelwise_mask(imgs)
        elif mask_type == 'proto':
            if local_iter <=20000:
                input_mask = self.generate_mask(imgs) 
            else:
                # mmcv.print_log(f'local_iter: {local_iter}', 'mmseg')
                input_mask = self.generate_proto_mask(imgs, proto, target_rep, local_iter)
        elif mask_type == 'proto_prob':
            if local_iter <=20000:
                input_mask = self.generate_mask(imgs) 
            else:
                # mmcv.print_log(f'local_iter: {local_iter}', 'mmseg')
                input_mask = self.generate_proto_prob_mask(imgs, proto, target_rep, local_iter)              
        return imgs * input_mask
