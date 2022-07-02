import argparse, sys, os, numpy as np, tqdm, torch, time, torch.optim as optim, gc
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./extrinsic_models/"))
from data_vp.dhf1k_dataset import Dhf1kDataset
from torch.utils.data import DataLoader
from mmv import vinet_plus as mmv_vinet
from model_vp import model as vinet_model
# from loss_vp import loss
from loss_vp import kldiv_loss
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Vinet Plus')
    parser.add_argument('--dataset_folder', "-d", required=True, type=str, help='Folder consisting of all the videos required for training purpose')
    parser.add_argument('--annotation_folder', "-a", required=True, type=str, help='Folder consisting of all the continuous saliency annotations for each video')
    parser.add_argument('--time_width', default=32, type=int, help='Time steps required for training and predicting the saliency map of last frame')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU for training')
    parser.add_argument('--epochs', default=50, type=int, help='Total epochs required for training')
    # Pytorch Distributed parameters
    parser.add_argument('--local_rank', type=int, help='Automatically passed by the PyTorch (Stores the rank of each process in the process group)')
    args = parser.parse_args()
    args.number_of_gpus = int(os.environ['WORLD_SIZE'])
    return args

# Copied from https://theaisummer.com/distributed-training-pytorch/
def init_distributed(args):
    dist.init_process_group(backend="nccl", init_method='env://', world_size=args.number_of_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.barrier()
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        if not is_master:
            f = open(os.devnull, 'w')
            sys.stdout = f

    setup_for_distributed(args.local_rank == 0)

# MMVN (Encodings are shifted to cuda)
def mmv_encodings(video_data):
    mmv_preprocessed_data = video_data.permute(0, 1, 3, 4, 2).numpy()
    # print(mmv_preprocessed_data.shape) # (batch_size, time_width, height, width, channels)
    mmv_embeddings = []
    for i in range(mmv_preprocessed_data.shape[0]):
        vid_repr = mmv_vinet.calculate_mmv_embeddings(mmv_preprocessed_data[[i]])
        mmv_embeddings.append(vid_repr)
    
    result = []
    for i in range(len(mmv_embeddings[0])): # Iterate over embedding layers
        result.append([])
        for j in range(len(mmv_embeddings)): # Iterate over batch size
            result[-1].append(mmv_embeddings[j][i])
        result[-1] = torch.stack(result[-1], dim = 0)
    
    return result

def vinet_preprocessings(video_data, vinet_mean, vinet_std):
    video_data = (video_data - vinet_mean) / vinet_std # mean-std normalization
    video_data = video_data.permute(0, 2, 1, 3, 4) # ready to be passed as input to ViNet
    # print(video_data.shape) # (batch_size, channels, time_width, height, width)
    return video_data

def load_vinet_encoder_weights(file_weight, model):
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight, map_location = next(model.parameters()).device)
        model_dict = model.backbone.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if 'base.' in name:
                bn = int(name.split('.')[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if bn >= sn_list[1] and bn < sn_list[2]:
                    sn = sn_list[1]
                elif bn >= sn_list[2] and bn < sn_list[3]:
                    sn = sn_list[2]
                elif bn >= sn_list[3]:
                    sn = sn_list[3]
                name = '.'.join(name.split('.')[2:])
                name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        print('loaded')
        model.backbone.load_state_dict(model_dict)
    else:
        print('weight file?')

if __name__ == '__main__':
    args = parse_args()
    init_distributed(args)

    # For mean-std normalization for ViNet preprocessings
    vinet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda().reshape(1, 1, 3, 1, 1)
    vinet_std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().reshape(1, 1, 3, 1, 1)
    # For resizing of ViNet preprocessings
    vinet_height, vinet_width = 288, 512 # 16:9 * 26

    model = vinet_model.VideoSaliencyModel(args.time_width).cuda()
    load_vinet_encoder_weights(os.path.join(os.path.dirname(vinet_model.__file__), 'S3D_kinetics400.pt'), model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = kldiv_loss.KLDLoss()
    optimizer = optim.Adam(model.parameters())

    # Create dataset and dataloader
    train_dataset = Dhf1kDataset(args.dataset_folder, args.annotation_folder, args.time_width, 'train', (vinet_height, vinet_width), args.local_rank == 0)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers = 1, sampler = train_sampler)
    validate_dataset = Dhf1kDataset(args.dataset_folder, args.annotation_folder, args.time_width, 'validate', (vinet_height, vinet_width), args.local_rank == 0)
    validate_sampler = DistributedSampler(dataset=validate_dataset)
    validate_dataloader = DataLoader(validate_dataset, batch_size = 1, num_workers = 1, sampler = validate_sampler)

    for epoch in range(args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for data in tqdm.tqdm(train_dataloader, desc = 'data iteration', disable=(args.local_rank != 0)):
            optimizer.zero_grad()

            video_data, saliency_map_gt = data
            # print(video_data.shape, saliency_map_gt.shape, video_data.dtype, saliency_map_gt.dtype, video_data.device, saliency_map_gt.device) # (batch_size, time_width, channels, height, width) (batch_size, 1, height, width) float32 float32 cpu cpu

            # Finding MMV embeddings
            mmv_embeddings = mmv_encodings(video_data)
            # print([r.shape for r in mmv_embeddings]) # Each element is of the shape (batch_size, channels, time_width, height, width)

            # ViNet preprocessings
            video_data, saliency_map_gt = video_data.cuda(), saliency_map_gt.cuda()
            video_data = vinet_preprocessings(video_data, vinet_mean, vinet_std)
            # print(video_data.shape, saliency_map_gt.shape, video_data.dtype, saliency_map_gt.dtype, video_data.device, saliency_map_gt.device) # (batch_size, channels, time_width, height, width) (batch_size, 1, height, width) float32 float32 cuda cuda

            pred_sal = model(video_data, mmv_embeddings)
            # print(f'Done {pred_sal.shape}') # (1, 1, height, width)

            pred_sal, saliency_map_gt = pred_sal[:, 0, :, :], saliency_map_gt[:, 0, :, :]

            loss = criterion(pred_sal, saliency_map_gt)
            loss.backward()
            optimizer.step()

            del video_data, saliency_map_gt, mmv_embeddings
            gc.collect()
            torch.cuda.empty_cache()

    dist.destroy_process_group() # cleanup