import argparse, os, torch.distributed as dist, torch, sys
from mmv import vinet_plus as mmv_vinet

DATASET_CONFIG = {
    "Dhf1k":{"height": 288, "width": 512}
}

def parse_args():
    parser = argparse.ArgumentParser(description='Vinet Plus')
    parser.add_argument('--dataset_folder', "-d", required=True, type=str, help='Folder consisting of all the videos required for training purpose')
    parser.add_argument('--annotation_folder', "-a", required=True, type=str, help='Folder consisting of all the continuous saliency annotations for each video')
    parser.add_argument('--log_dir', '-l', required=False, type=str, default=f'/ssd_scratch/cvit/{os.getlogin()}/logs/', help='Folder to log the outputs')
    parser.add_argument('--checkpoint_path', '-c', required=False, type=str, default='', help='Path to the weights_<epoch>.pth file')
    parser.add_argument('--time_width', default=32, type=int, help='Time steps required for training and predicting the saliency map of last frame')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU for training')
    parser.add_argument('--epochs', default=200, type=int, help='Total epochs required for training')
    # Pytorch Distributed parameters
    parser.add_argument('--local_rank', type=int, help='Automatically passed by the PyTorch (Stores the rank of each process in the process group)')
    args = parser.parse_args()
    args.number_of_gpus = int(os.environ['WORLD_SIZE'])
    return args

# Copied from https://theaisummer.com/distributed-training-pytorch/
def init_distributed(args):
    dist.init_process_group(backend="nccl", init_method='env://', world_size=args.number_of_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        if not is_master:
            f = open(os.devnull, 'w')
            sys.stdout = f

    setup_for_distributed(args.local_rank == 0)
    dist.barrier() # Sync all the processes at this line (faster processes stop till all the processes can continue together)

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

