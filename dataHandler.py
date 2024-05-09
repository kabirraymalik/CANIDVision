import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np

class FrameDataset(Dataset):
    def __init__(self, numFrames, transform=None):
        self.root_dir = os.path.join(os.getcwd()+"/data/")
        # Specify transformations
        self.transform = transform

        if self._genFrames(numFrames):
            print("frames parsed successfully.")
        else: 
            print("didn't parse video data.")

        self.labels = os.listdir(self.root_dir+"video_data/")
        self.labels.sort()
        self.label_to_idx = {label_name: i for i, label_name in enumerate(self.labels)}
        self.videos = self._load_videos()

    def _genFrames(self, numFrames):
        path = os.getcwd()
        if os.path.exists(path + "/data/build/") == False:
            os.mkdir(path+"/data/build/")
        else:
            print("/data/build already exists!")
            return 0
        videoData = os.fsencode(path+"/data/video_data/")
        for labelobj in os.listdir(videoData):
            label = str(labelobj)[2:-1]
            if label.isalnum():
                os.mkdir(path+"/data/build/"+label+"/")
                for video in os.listdir(path+"/data/video_data/"+label):
                    dirName = path+"/data/build/" + label + "/" + str(video)[0:-4]
                    os.mkdir(dirName)
                    probe = ffmpeg.probe(path+"/data/video_data/"+label+"/"+str(video))
                    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
                    duration = float(video_info['duration'])
                    # Calculate intervals for frame extraction
                    intervals = np.linspace(0, duration, numFrames, endpoint=False)
                    for i, interval in enumerate(intervals):
                        output_file = f"frame_{i+1}.bmp"  # Define output frame filename
                        # Extract frame
                        (
                        ffmpeg
                        .input(path+"/data/video_data/"+label+"/"+str(video), ss=interval)  # Seek to position
                        .output(path+"/data/build/"+label+"/"+str(video)[0:-4]+"/"+output_file, vframes=1)  # Extract one frame
                        .run(capture_stdout=True, capture_stderr=True)
                        )
            progress = (os.listdir(videoData).index(labelobj)+1)/len(os.listdir(videoData)) * 100
            print("building... " + str(round(progress,4)) + "% complete")
        return 1

    def _load_videos(self):
        vid_paths = []  # Renamed for clarity
        vid_labels = []  # Stores the label indices corresponding to each frame
        image_data_dir = os.getcwd()+'/data/build/'
        for label in self.labels:
            if str(label).isalnum():
                label_dir = os.path.join(image_data_dir, label)
                for video in os.listdir(label_dir):
                    video_dir = os.path.join(label_dir, video)
                    vid_paths.append(video_dir)
                    vid_labels.append(self.label_to_idx[label])
        return list(zip(vid_paths, vid_labels))

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        image_tensors = []
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            image = Image.open(frame_path).convert('RGB')  # Convert to RGB for model compatibility
            if self.transform:
                image = self.transform(image) #applies transform
            image_tensors.append(image)
        sequence_tensor = torch.stack(image_tensors, dim=0)
        return sequence_tensor, label

    def __len__(self):
        return len(self.videos)
