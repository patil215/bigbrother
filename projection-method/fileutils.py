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

def read_video_frames(source, startIndex, endIndex):
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

def write_video_frames(video_source, startIndex, endIndex, filename):
    videoFrames = read_video_frames(video_source, startIndex, endIndex)
    write_obj(filename, videoFrames)
    print("[{0} - {1}] {2} frame segment saved successfully to {3}".format(startIndex, endIndex, len(videoFrames), filename))

def read_training_data(data_dir):
    file_tree = get_file_tree(data_dir)
    for subfolder in file_tree:
        file_tree[subfolder] = [read_obj("{}/{}/{}".format(data_dir, subfolder, subfile)) 
                                for subfile in file_tree[subfolder]]
    return file_tree

def get_file_tree(root_dir):
    """
    Returns a dictionary with the keys as subfolders and values as a set of filenames.
    Only works for root directories with one level of child folders.
    """
    data_dict = {}
    subfolders = [f.name for f in os.scandir(root_dir) if f.is_dir()]

    for subfolder in subfolders:
        data_dict[subfolder] = set([f.name for f in os.scandir(root_dir + '/' + subfolder) if f.is_file()])

    return data_dict

def get_test_segment_tree(root_dir):
    tree = get_file_tree(root_dir)
    for folder in tree:
        tree[folder] = list(filter(lambda filename: filename[0] != '.', tree[folder]))
    return tree

def get_test_path_tree(root_dir):
    tree = get_file_tree(root_dir)
    for folder in tree:
        tree[folder] = list(filter(lambda filename: filename[0] == '.' and filename[-5:] == ".path", tree[folder]))
    return tree
