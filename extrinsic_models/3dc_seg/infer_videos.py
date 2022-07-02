import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='SaliencySegmentation')
    parser.add_argument('--videos_path', "-v", required=True, type=str)
    parser.add_argument("--results_dir", "-r", required=True, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for video_file in os.listdir(args.videos_path):
        video_file = os.path.join(args.videos_path, video_file)
        video_basename = os.path.splitext(os.path.basename(video_file))[0]

        os.system(f'python infer_video.py -c run_configs/bmvc_final_dense.yaml --wts ./bmvc_final.pth --video_file {video_file} --results_dir {os.path.join(args.results_dir, video_basename)}')