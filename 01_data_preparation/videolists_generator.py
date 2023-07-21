import argparse
import os
import pandas as pd
import math

def parse_args():
    parser = argparse.ArgumentParser(
        description='make video list for skeleton extraction')
    parser.add_argument('--dir', type=str, help='path for raw video directory')
    parser.add_argument('--move', type=str, help='the kind of movement')
    parser.add_argument('--eval', type=str, help='path for evaluation data file')
    args = parser.parse_args()
    return args
    
def get_label_from_eval_csv(eval_file, subject):
    df = pd.read_csv(eval_file) 
    # print(df)
    subject_column = 'Subject' 
    label_column = 'Binary'
    filtered_df = df[df[subject_column] == subject]
    label = filtered_df[label_column].values[0] if len(filtered_df) > 0 else None
    if not math.isnan(label):
        label = int(label)
    return label

def save_to_txt(video_path, name, label, move):
    txt_file = f'p_data/video_lists/{move}/' + name + '.txt'
    if label is not None:
        with open(txt_file, 'w') as f:
            f.write(f"{video_path} {label}")
    else: 
        with open(txt_file, 'w') as f:
            f.write(f"{video_path}")

def main():
    args = parse_args()
    move = args.move
    path = args.dir + '/' + move

    if args.eval is not None:
        for root, dirs, files in os.walk(path):
            for file in files:
                subject = file.split('_')[1]  
                video_path = os.path.join(root, file)
                name = file[:-4]
                label = get_label_from_eval_csv(args.eval, subject)
                print(label)
                if len(str(label)) > 1:
                    print(video_path)
                    continue
                save_to_txt(video_path, name, label, move)

    # if there is no label
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                subject = file.split('_')[1]  
                video_path = os.path.join(root, file)
                name = file[:-4]
                save_to_txt(video_path, name, None, move)

if __name__ == '__main__':
    main()
