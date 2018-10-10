import pickle
import os
from collections import defaultdict

def read_obj(path):
	return pickle.load(open(path, 'rb'))

def write_obj(path, object):
    with open(path, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

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