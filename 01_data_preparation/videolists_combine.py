import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='combined video list for skeleton extraction')
    parser.add_argument('--dir', type=str, help='path for videolists')
    parser.add_argument('--move', type=str, help='the kind of movement')
    parser.add_argument('--out', type=str, help='path for combined text')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    move = args.move
    path = args.dir + '/' + move
    directory = f'p_data/video_lists/{move}/'
    output_file = f'p_data/video_lists_combined/{move}.txt'

    file_list = [file for file in os.listdir(directory) if file.endswith('.txt')]
    file_list.sort()
    print(file_list)

    with open(output_file, 'a') as outfile:
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as infile:
                lines = infile.read().splitlines()
                outfile.write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()