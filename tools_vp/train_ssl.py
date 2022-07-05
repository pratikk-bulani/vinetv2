import sys, os, numpy as np, tqdm, torch, time, torch.optim as optim, gc, cv2
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./extrinsic_models/"))
from data_vp.dhf1k_dataset import Dhf1kDataset
from torch.utils.data import DataLoader
from model_vp import model as vinet_model
# from loss_vp import loss
from loss_vp import kldiv_loss
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import utils

def train_one_epoch(epoch, train_dataloader, model, optimizer, criterion, args):
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    total_loss = 0

    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        video_data, saliency_map_gt = data
        # print(video_data.shape, saliency_map_gt.shape, video_data.dtype, saliency_map_gt.dtype, video_data.device, saliency_map_gt.device) # (batch_size, time_width, channels, height, width) (batch_size, 1, height, width) float32 float32 cpu cpu

        # Finding MMV embeddings
        mmv_embeddings = utils.mmv_encodings(video_data)
        # print([r.shape for r in mmv_embeddings]) # Each element is of the shape (batch_size, channels, time_width, height, width)

        del video_data
        saliency_map_gt = saliency_map_gt.cuda()
        # print(saliency_map_gt.shape, saliency_map_gt.dtype, saliency_map_gt.device) # (batch_size, 1, height, width) float32 cuda

        pred_sal = model(mmv_embeddings).view(args.batch_size, utils.DATASET_CONFIG['Dhf1k']['height'], utils.DATASET_CONFIG['Dhf1k']['width'])
        # print(f'Done {pred_sal.shape}') # (1, height, width)

        loss = criterion(pred_sal, saliency_map_gt.view(args.batch_size, utils.DATASET_CONFIG['Dhf1k']['height'], utils.DATASET_CONFIG['Dhf1k']['width']))
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        total_loss += loss.item()
        
        if args.local_rank == 0 and i == 0:
            summary.add_image('Training Saliency/GT', saliency_map_gt[0, Ellipsis].cpu().numpy(), global_step=epoch)
            summary.add_image('Training Saliency/Pred', pred_sal.detach().cpu().numpy(), global_step=epoch)

        del saliency_map_gt, mmv_embeddings; gc.collect(); torch.cuda.empty_cache();
    return total_loss

if __name__ == '__main__':
    args = utils.parse_args()
    utils.init_distributed(args)
    if args.checkpoint_path == "":
        args.log_dir = os.path.join(args.log_dir, datetime.now().strftime(r"%Y-%m-%d#%H-%M-%S"))
        os.makedirs(args.log_dir, exist_ok=True)
    else:
        args.log_dir = os.path.dirname(args.checkpoint_path)

    model = vinet_model.SSLOnly().cuda()
    start_epoch = 0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = optim.Adam(model.parameters())
    criterion = kldiv_loss.kldiv
    # criterion = torch.nn.MSELoss()

    # Create dataset and dataloader
    train_dataset = Dhf1kDataset(args.dataset_folder, args.annotation_folder, args.time_width, 'train', (utils.DATASET_CONFIG['Dhf1k']['height'], utils.DATASET_CONFIG['Dhf1k']['width']), args.local_rank == 0)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 1, sampler = train_sampler)
    validate_dataset = Dhf1kDataset(args.dataset_folder, args.annotation_folder, args.time_width, 'validate', (utils.DATASET_CONFIG['Dhf1k']['height'], utils.DATASET_CONFIG['Dhf1k']['width']), args.local_rank == 0)
    validate_sampler = DistributedSampler(dataset=validate_dataset)
    validate_dataloader = DataLoader(validate_dataset, batch_size = 1, num_workers = 1, sampler = validate_sampler)

    if args.local_rank == 0:
        summary = SummaryWriter(log_dir=args.log_dir, flush_secs=1)
    
    for epoch in tqdm.tqdm(range(start_epoch, args.epochs), desc = 'Epoch iteration', disable=(args.local_rank != 0)):
        dist.barrier()
        loss = train_one_epoch(epoch, train_dataloader, model, optimizer, criterion, args)
        if args.local_rank == 0:
            summary.add_scalar("Training Loss", loss, global_step=epoch)

    dist.destroy_process_group() # cleanup