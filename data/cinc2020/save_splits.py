import numpy as np, os, sys, joblib
from scipy.io import loadmat
from glob import iglob

    
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)
    
if __name__ == "__main__":
    input_directory = "/home/gridsan/ybo/advaug/data/cinc2020/raw/training"
    header_files = []

    level3 = iglob("/home/gridsan/ybo/advaug/data/cinc2020/raw/training/*/*/*")
    for f in level3:
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(f):
            header_files.append(f)

    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    num_files = len(header_files)

    recordings = list()
    headers = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording)
        headers.append(header)

    recordings = np.array(recordings)
    headers = np.array(headers)

    recordings.dump("recordings.npy", protocol=4)
    headers.dump("headers.npy", protocol=4)


    