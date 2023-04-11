import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video-list', type=str, help='the list of source videos')
parser.add_argument('--out', type=str, help='output pickle name')
args = parser.parse_args()

with open(args.video_list, 'r') as f:
    video_list = f.readlines()

data_dict = {}
for line in video_list:
    parts = line.strip().split()
    if len(parts) == 1:
        data_dict[parts[0]] = None
    elif len(parts) == 2:
        data_dict[parts[0]] = int(parts[1])
    else:
        raise ValueError('Invalid video list format')

if not args.out.endswith('.pkl'):
    args.out += '.pkl'
with open(args.out, 'wb') as f:
    pickle.dump(data_dict, f)
