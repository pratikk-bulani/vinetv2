import argparse
from utils.util import synchronize, cleanup_env, get_optimiser, get_model, _find_free_port
from config import get_cfg
from datasets.davis.Davis import Davis
import os, cv2
import shutil
import glob
from datasets.BaseDataset import INFO, IMAGES_
import numpy as np
from PIL import Image
from utils.Resize import resize
import apex
import torch
from apex import amp
from utils.Saver import load_weightsV2
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from cv2 import resize as imresize
from util import color_map

def init_torch_distributed(port):
  print("devices available: {}".format(torch.cuda.device_count()))
  #port = _find_free_port()
  print("Using port {} for torch distributed.".format(port))
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
      'nccl',
      init_method='env://',
    )
  else:
    dist_url = "tcp://127.0.0.1:{}".format(port)
    try:
      dist.init_process_group(
        backend="NCCL",
        init_method=dist_url, world_size=1, rank=0
      )
    except Exception as e:
      print("Process group URL: {}".format(dist_url))
      raise e

class InferenceDataset(Davis):
    def __init__(self, video_file, results_dir, cfg):
        self.video_file = video_file
        self.results_dir = results_dir
        self.frames_dir = os.path.join(results_dir, "frames") # directory to store all the frames of the given video
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        os.system(f'ffmpeg -i {self.video_file} {os.path.join(self.frames_dir, "%04d.png")}') # Extracting all the frames of a video
        super(InferenceDataset, self).__init__(root = self.frames_dir, mode = 'test', resize_mode = cfg.INPUT.RESIZE_MODE_TEST, resize_shape = cfg.INPUT.RESIZE_SHAPE_TEST, tw = 20, max_temporal_gap = cfg.DATASETS.MAX_TEMPORAL_GAP, imset = cfg.DATASETS.IMSET)
    
    def get_support_indices(self, index):
        index_range = np.arange(index, min(self.num_frames, (index + self.tw)))
        support_indices = np.random.choice(index_range, min(self.tw, len(index_range)), replace=False)
        support_indices = np.sort(np.append(support_indices, np.repeat([index], self.tw - len(support_indices))))
        return support_indices
    
    def create_sample_list(self):
        img_list = sorted(list(glob.glob(os.path.join(self.frames_dir, '*.png'))))[:600] # stores all the images of a video
        self.num_frames = len(img_list)
        self.num_objects = 1
        self.shape = np.shape(Image.open(os.path.join(self.frames_dir, '0001.png')))

        for i, img in enumerate(img_list):
            sample = {INFO: {}, IMAGES_: []}
            support_indices = self.get_support_indices(i)
            sample[INFO]['support_indices'] = support_indices
            images = [os.path.join(self.frames_dir, '{:04d}.png'.format(s+1)) for s in support_indices]
            sample[IMAGES_] = images

            sample[INFO]['num_frames'] = self.num_frames
            sample[INFO]['num_objects'] = self.num_objects
            sample[INFO]['shape'] = self.shape

            self.samples+=[sample]
    
    def read_sample(self, sample):
        images = self.read_image(sample)

        images_resized = []
        for im in images:
            data = {"image": im}
            data = resize(data, self.resize_mode, self.resize_shape)
            images_resized += [data['image']]

        images = np.stack(images_resized)

        data = {IMAGES_: images}
        for key, val in sample.items():
            if key in ['images']:
                continue
            if key in data:
                data[key] += [val]
            else:
                data[key] = [val]
        return data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensors_resized = self.read_sample(sample)

        padded_tensors = self.pad_tensors(tensors_resized)

        padded_tensors = self.normalise(padded_tensors)

        return {"images": np.transpose(padded_tensors['images'], (3, 0, 1, 2)).astype(np.float32),
            'info': padded_tensors['info']}
    
    def pad_tensors(self, tensors_resized):
        h, w = tensors_resized["images"].shape[1:3]
        new_h = h + 32 - h % 32 if h % 32 > 0 else h
        new_w = w + 32 - w % 32 if w % 32 > 0 else w
        lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
        lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)

        padded_tensors = tensors_resized.copy()
        keys = ['images']

        for key in keys:
            pt = []
            t = tensors_resized[key]
            if t.ndim == 3:
                t = t[..., None]
            assert t.ndim == 4
            padded_tensors[key] = np.pad(t, ((0,0),(lh, uh), (lw, uw), (0, 0)), mode='constant')

        padded_tensors['info'][0]['pad'] = ((lh, uh), (lw, uw))
        return padded_tensors

def parse_args():
    parser = argparse.ArgumentParser(description='SaliencySegmentation')
    parser.add_argument('--config', "-c", required=True, type=str)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=4, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--print_freq', dest='print_freq',
                        help='Frequency of statistics printing',
                        default=1, type=int)
    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                        help='epoch to load model',
                        default=None, type=str)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='load pretrained weights for PWCNet',
                        default='weights/pwc_net.pth.tar', type=str)
    parser.add_argument('--wts', '-w', dest='wts',
                        help='weights file to resume training',
                        default=None, type=str)
    parser.add_argument('--video_file', "-v", required=True, type=str, help='Video to Infer')
    parser.add_argument('--results_dir', "-r", required=True, type=str, help='Store results to')

    # summary generation
    parser.add_argument('--show_image_summary', dest='show_image_summary',
                        help='load the best model',
                        default=False, type=bool)
    args = parser.parse_args()
    return args

class Inferencer:
    def __init__(self, args, port):
        cfg = get_cfg()
        self.port = port
        cfg.merge_from_file(args.config)
        self.cfg = cfg
        self.results_dir = args.results_dir
        self.video_file = args.video_file
        self.model = get_model(cfg)
        self.model_dir = None
        self.model, self.optimiser = self.init_distributed(cfg)
        self.testset = InferenceDataset(args.video_file, args.results_dir, cfg)
    def init_distributed(self, cfg):
        torch.cuda.set_device(args.local_rank)
        init_torch_distributed(self.port)
        model = apex.parallel.convert_syncbn_model(self.model)
        model.cuda()
        optimiser = get_optimiser(model, cfg)
        model, optimiser, self.start_epoch, self.iteration = \
        load_weightsV2(model, optimiser, args.wts, self.model_dir)
        # model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp_weights = \
        #   load_weights(model, self.optimiser, args, self.model_dir, scheduler=None, amp=amp)  # params
        # lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)
        opt_levels = {'fp32': 'O0', 'fp16': 'O2', 'mixed': 'O1'}
        if cfg.TRAINING.PRECISION in opt_levels:
            opt_level = opt_levels[cfg.TRAINING.PRECISION]
        else:
            opt_level = opt_levels['fp32']
            print('WARN: Precision string is not understood. Falling back to fp32')
        model, optimiser = amp.initialize(model, optimiser, opt_level=opt_level)
        # amp.load_state_dict(amp_weights)
        if torch.cuda.device_count() > 1:
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        self.world_size = torch.distributed.get_world_size()
        print("Intitialised distributed with world size {} and rank {}".format(self.world_size, args.local_rank))
        return model, optimiser
    def start(self):
        inference_engine = SaliencyInferenceEngine(self.cfg, self.results_dir)
        inference_engine.infer(self.testset, self.model)
        shutil.rmtree(self.testset.frames_dir, ignore_errors = True)

class SaliencyInferenceEngine:
    def __init__(self, cfg, results_dir) -> None:
        self.cfg = cfg
        self.results_dir = results_dir
    def infer(self, dataset, model):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, sampler=None, pin_memory=True)
        all_semantic_pred = {}
        with torch.no_grad():
            for iter, input_dict in tqdm(enumerate(dataloader)):
                if not self.cfg.INFERENCE.EXHAUSTIVE and (iter % (self.cfg.INPUT.TW - self.cfg.INFERENCE.CLIP_OVERLAP)) != 0:
                    continue
                info = input_dict['info'][0]
                input_var = input_dict["images"]
                batch_size = input_var.shape[0]
                input_var = input_var.float().cuda()
                pred = model(input_var)
                pred_mask = F.softmax(pred[0], dim=1)
                clip_frames = info['support_indices'][0].data.cpu().numpy()
                assert batch_size == 1
                for i, f in enumerate(clip_frames):
                    if f in all_semantic_pred:
                        all_semantic_pred[f] += [pred_mask[0, :, i].data.cpu().float()]
                    else:
                        all_semantic_pred[f] = [pred_mask[0, :, i].data.cpu().float()]
            self.save_results(all_semantic_pred, info)
    def save_results(self, pred, info):
        results_path = os.path.join(self.results_dir, 'predictions')
        print(results_path)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        (lh, uh), (lw, uw) = info['pad']
        for f in pred.keys():
            M = torch.argmax(torch.stack(pred[f]).mean(dim=0), dim=0)
            h, w = M.shape[-2:]
            M = M[lh[0]:h - uh[0], lw[0]:w - uw[0]]
            shape = info['shape']
            # print(type(M.byte()), type(M), shape[0].item(), type(M.numpy()), M.shape) # torch.Tensor, torch.Tensor, [360, 640, 3], numpy.array
            # img_M = Image.fromarray(imresize(M.byte(), shape, interp='nearest')) # Earlier
            img_M = Image.fromarray(imresize(M.numpy(), [shape[1].item(), shape[0].item()], interpolation=cv2.INTER_NEAREST)).convert('P')
            img_M.putpalette(color_map().flatten().tolist())
            img_M.save(os.path.join(results_path, '{:04d}.png'.format(f+1)))
            

if __name__ == '__main__':
    args = parse_args()
    port = _find_free_port()
    shutil.rmtree(args.results_dir, ignore_errors = True)
    os.makedirs(args.results_dir)
    inferencer = Inferencer(args, port)
    inferencer.start()
    synchronize()
    cleanup_env()