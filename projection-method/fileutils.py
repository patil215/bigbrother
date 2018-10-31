import pickle
import os
from collections import defaultdict
import cv2

def read_obj(path):
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, 'rb'))

def write_obj(path, object):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def load_video_clip(source, startIndex, endIndex):
    """
    Load raw frames of a video source into a list, from [startIndex, endIndex], inclusive.
    """
    video = cv2.VideoCapture(source)
    videoFrames = []

    for index in range(startIndex, endIndex + 1):
        video.set(1, index)
        ok, rawFrame = video.read()
        videoFrames.append(rawFrame)

    return videoFrames

def save_video_clip(video_source, startIndex, endIndex, filename):
    videoFrames = load_video_clip(video_source, startIndex, endIndex)
    write_obj(filename, videoFrames)
    print("[{0} - {1}] {2} frame segment saved successfully to {3}".format(startIndex, endIndex, len(videoFrames), filename))

def readData(data_dir):
    data_dict = defaultdict(list)

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir() ]    

    for subfolder in subfolders:
        fullpath = data_dir + '/' + subfolder

        subfiles = [f.name for f in os.scandir(fullpath) if f.is_file() ]
        for subfile in subfiles:
            fullest_path = fullpath + '/' + subfile
            data_dict[subfolder].append(read_obj(fullest_path))

    return data_dict

def readVideos(data_dir):
    data_dict = defaultdict(dict)

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir() ]    

    for subfolder in subfolders:
        fullpath = data_dir + '/' + subfolder

        subfiles = [f.name for f in os.scandir(fullpath) if f.is_file() ]
        for subfile in subfiles:
            if subfile[0] == '.':
                continue
            fullest_path = fullpath + '/' + subfile
            data_dict[subfolder][subfile] = read_obj(fullest_path)

    return data_dict