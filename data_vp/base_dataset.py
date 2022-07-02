import torch, os

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, annotation_folder, time_width, mode, resize_shape, is_master):
        self.dataset_folder = os.path.join(dataset_folder, mode)
        self.annotation_folder = os.path.join(annotation_folder, mode)
        self.time_width = time_width
        assert mode in ['train', 'validate', 'test']
        self.mode = mode
        self.resize_shape = resize_shape
        self.is_master = is_master
