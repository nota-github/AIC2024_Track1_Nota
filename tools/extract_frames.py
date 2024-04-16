import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

def make_parser():
    parser = argparse.ArgumentParser(description="Extract Frames from AIC24 Track1 Videos")
    parser.add_argument("-p", "--path", type=str, default=None, help="path to AIC24_Track1_MTMC_Tracking folder")
    parser.add_argument("-n", "--num-processes", type=int, default=4, help="number of processes to use")
    return parser.parse_args()

def extract_frames(video_file, output_folder, file_headname):
    # Use ffmpeg to extract frames from the video file
    command = [
        'ffmpeg',
        '-ss', '00:00:0',
        '-i', video_file,
        '-r', '30',
        '-q', '2',
        '-f', 'image2',
        os.path.join(output_folder, f"{file_headname}_%d.jpg")
    ]
    subprocess.run(command)

def process_video(vid, path):
    output_folder = f"./data/frames/{vid.parts[-4]}/{vid.parts[-3]}/{vid.parts[-2]}"
    os.makedirs(output_folder, exist_ok=True)
    file_headname = vid.parts[-3] + vid.parts[-2]
    extract_frames(str(vid), output_folder, file_headname)

def main(path, num_processes):
    videos = list(Path(path).glob('**/*.mp4'))  # Convert generator to list to use with Executor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for _ in tqdm(executor.map(process_video, videos, [path]*len(videos)), total=len(videos)):
            pass

if __name__ == "__main__":
    args = make_parser()
    main(args.path, args.num_processes)