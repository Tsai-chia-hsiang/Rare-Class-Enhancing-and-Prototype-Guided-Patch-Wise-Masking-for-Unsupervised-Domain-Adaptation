import torch
import gc
import mmcv

class MemCache:

    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024 ** 2), 3)

    def __init__(self):
        self.dctn = {}
        self.max_reserved = 0
        self.max_allocate = 0

    def mclean(self):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in list(self.dctn.keys()):
            del self.dctn[key]
        gc.collect()
        torch.cuda.empty_cache()

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Mem Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.dctn[key] = value
        self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(0))
        self.max_allocate = max(self.max_allocate, torch.cuda.memory_allocated(0))

    def __getitem__(self, item):
        return self.dctn[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in keys:
            del self.dctn[key]

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Cuda Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a
        
        mmcv.print_log('Cuda Info', 'mmseg')
        mmcv.print_log(f'Total     \t{MemCache.byte2MB(t)} MB', 'mmseg')
        mmcv.print_log(f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(self.max_reserved)}] MB', 'mmseg')
        mmcv.print_log(f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(self.max_allocate)}] MB', 'mmseg')
        mmcv.print_log(f'Free      \t{MemCache.byte2MB(f)} MB', 'mmseg')

        # print('Cuda Info')
        # print(f'Total     \t{MemCache.byte2MB(t)} MB')
        # print(f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(self.max_reserved)}] MB')
        # print(f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(self.max_allocate)}] MB')
        # print(f'Free      \t{MemCache.byte2MB(f)} MB')