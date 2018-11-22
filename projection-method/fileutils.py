import pickle
import os
from collections import defaultdict
import cv2

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def read_obj(path):
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, 'rb'))

def write_obj(path, object):
    if path.rfind(".") == -1:
        print("Warning: saving file without an extension: {}".format(path))

    make_dir(os.path.dirname(path))

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

def get_paths(path_dir):
    paths = []
    file_tree = get_file_tree(path_dir)
    for subfolder in file_tree:
        other_paths = get_file_tree(path_dir + '/' + subfolder)
        for thing in other_paths:
            set_things = other_paths[thing]
            for item in set_things:
                if ".path" in item:
                    string = path_dir + '/' + subfolder + '/' + thing + '/' + item
                    paths.append(string)
    return paths

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
        tree[folder] = list(filter(lambda filename: filename[0] != '.' and filename[-5:] != ".path", tree[folder]))
    return tree

def get_test_path_tree(root_dir):
    tree = get_file_tree(root_dir)
    for folder in tree:
        tree[folder] = list(filter(lambda filename: filename[-5:] == ".path", tree[folder]))
    return tree

def get_next_file_number(root_dir):
    """
    Reads a directory of files with integer filenames and returns the next available file name
    (largest filename + 1), extension not included. Creates the directory if it does not exist.
    """
    make_dir(root_dir)
    base_file_nums = sorted([int(f.name.split('.')[0]) for f in os.scandir(root_dir) if f.is_file()])
    if len(base_file_nums) == 0:
        return 1 

    return base_file_nums[-1] + 1
