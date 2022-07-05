import os, imageio, numpy as np, tqdm, random, torch, cv2
from .base_dataset import BaseDataset
# from mmv import vinet_plus as mmv_vinet
from torchvision import transforms

class Dhf1kDataset(BaseDataset):
    def __init__(self, dataset_folder, annotation_folder, time_width, mode, resize_shape, is_master):
        super().__init__(dataset_folder, annotation_folder, time_width, mode, resize_shape, is_master)
        
        # Find the number of frames for each video (without having to load the entire video)
        self.all_videos = sorted(os.listdir(self.dataset_folder))[:10] # all the videos present in the dataset_folder
        self.total_videos = len(self.all_videos) # stores the total number of videos present in the dataset_folder
        self.frames_count = [] # stores the total frames present in each and every video
        for path in tqdm.tqdm(self.all_videos, desc=f'Loading {self.mode} Videos', disable=(not self.is_master)):
            video_path = os.path.join(self.dataset_folder, path)
            
            # Using ImageIO get_reader
            # video_data = imageio.get_reader(video_path, 'ffmpeg')
            # self.frames_count.append(video_data.count_frames())
            # video_data.close()

            # Using cv2 VideoCapture
            video_data = cv2.VideoCapture(video_path)
            self.frames_count.append(int(video_data.get(cv2.CAP_PROP_FRAME_COUNT)))
            video_data.release()

        self.frames_count = np.array(self.frames_count, dtype=np.int16)

        # Total dataset length (As every window of size = self.time_width can be our training data)
        # self.total_dataset_len = np.sum(self.frames_count - self.time_width + 1)

        # Image transform
        self.img_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize(self.resize_shape)
		])

    def __len__(self):
        # return self.total_dataset_len
        return len(self.all_videos)
    
    def __getitem__(self, idx):
        idx = idx % self.total_videos # Choosing a video
        video_selected = self.all_videos[idx] # Selected video
        frame_start_idx = random.randint(a = 0, b = self.frames_count[idx] - self.time_width) # fetching time_width frames starting from this index

        # Fetch the frames
        
        # Using ImageIO get_reader
        # video_data = imageio.get_reader(os.path.join(self.dataset_folder, video_selected), 'ffmpeg') # Load the video (without loading the entire video)
        # inter_result = []
        # for i in range(frame_start_idx, frame_start_idx + self.time_width):
        #     inter_result.append(self.img_transform(video_data.get_data(i)))
        # inter_result = torch.stack(inter_result, dim=0)
        # video_data.close()

        # Using cv2 VideoCapture
        video_data = cv2.VideoCapture(os.path.join(self.dataset_folder, video_selected))
        inter_result = []
        video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_start_idx)
        for i in range(self.time_width):
            res, frame = video_data.read()
            if not res: raise Exception("Video not read properly")
            inter_result.append(self.img_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_data.release()
        inter_result = torch.stack(inter_result, dim=0)
        # print(inter_result.shape, inter_result.dtype) # (time_width, channels, height, width) torch.float32

        if self.mode == 'test': return inter_result

        saliency_map = np.asarray(imageio.imread(os.path.join(self.annotation_folder, f"0{os.path.splitext(video_selected)[0]}", "maps", "{0:04d}.png".format(i+1)))) # Stores the saliency map of the last frame of the inter_result
        saliency_map = self.img_transform(saliency_map)
        # print(saliency_map.shape, saliency_map.max()) # (1, height, width) 0.9991
        
        return inter_result, saliency_map

        # The below code gives memory error
        # MMVN
        # preprocessed_data = inter_result[np.newaxis, ...].astype(np.float32) / 255.
        # vid_repr = mmv_vinet.calculate_mmv_embeddings(preprocessed_data)

        # results = {"mmv": vid_repr}
        # return vid_repr
