import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import mmcv
import gc

class prototype_dist_estimator():
    def __init__(self, feature_num, cfg):
        super(prototype_dist_estimator, self).__init__()

        self.cfg = cfg
        self.class_num = cfg['class_num']
        self.feature_num = 256 
        # momentum 
        self.momentum = cfg['momentum']

        # init prototype
        self.init(feature_num=feature_num)

    def init(self, feature_num):
    
        self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    @torch.no_grad()
    def front_update(self, features, labels):
        mask = (labels != 255)
        # mmcv.print_log(f'iter: {iter}', 'mmseg') 
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        
        N, A = features.size()
        C = self.class_num
        # refer to SDCA for fast implementation
        features = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = features.mul(NxCxA_onehot)
        Amount_CXA = NxCxA_onehot.sum(0)
        Amount_CXA[Amount_CXA == 0] = 1
        mean = features_by_sort.sum(0) / Amount_CXA
        sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
        weight = sum_weight.div(
            sum_weight + self.Amount.view(C, 1).expand(C, A)
        )
        weight[sum_weight == 0] = 0
        self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
        self.Amount = self.Amount + onehot.sum(0)
        gc.collect()
        torch.cuda.empty_cache()
        
    @torch.no_grad()   
    def later_update(self, source_features, source_labels, target_features, target_labels):
            mask = (source_labels != 255)
            # remove IGNORE_LABEL pixels
            source_labels = source_labels[mask]
            target_labels = target_labels[mask]
            source_features =  source_features[mask]
            target_features =  target_features[mask]
            # momentum implementation
            ids_unique = source_labels.unique()
            for i in ids_unique:
                i = i.item()
                s_mask_i = (source_labels == i)
                source_feature = source_features[s_mask_i]
                t_mask_i = (target_labels == i)
                target_feature = target_features[t_mask_i]
                all_feature = torch.cat((source_feature, target_feature), dim=0)
                feature = torch.mean(all_feature, dim=0)
                self.Amount[i] += len(s_mask_i)
                self.Amount[i] += len(t_mask_i)
                self.Proto[i, :] = (1 - self.momentum) * feature + self.Proto[i, :] * self.momentum
            gc.collect()
            torch.cuda.empty_cache()
        
        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name))
