import pickle

def read_obj(path):
	return pickle.load(open(path, 'rb'))

def write_obj(path, object):
    with open(path, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)