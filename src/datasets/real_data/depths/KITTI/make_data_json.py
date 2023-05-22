import json
import os
import re

from helper import create_folder_if_it_doesnt_exist


def atoi(text):
    try:
        return float(text) if '.' in text else int(text) if text.isdigit() else text
    except:
        return text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+\.?\d*)', text)]

"""
creating an annotation json file from the existing data.
"""
if __name__ == '__main__':
    annotation = []
    base_path = '.'  # 'datasets/real_data/depths/KITTI'
    start_path = f'{base_path}/rgbs'

    def recursive(path):
        global images
        subdirs = os.listdir(path)
        subdirs.sort(key=natural_keys)
        for dir in subdirs:
            if 'image_0' in dir and dir != 'image_03':
                continue
            curr_path = f"{path}/{dir}"
            if os.path.isdir(curr_path):
                recursive(curr_path)
            elif os.path.isfile(curr_path):
                try:
                    rgb_path = curr_path
                    folder_name = curr_path.split("/")[6]
                    file_name = curr_path.split("/")[-1]
                    if file_name.split(".")[-1] != 'png':
                        continue
                    depth_path = f"{base_path}/full_depths/{folder_name}/proj_depth/groundtruth/image_03/{file_name}"
                    if not os.path.isfile(depth_path):
                        print("does not exist")
                        continue
                    global annotation
                    annotation += [{"rgb_path": rgb_path.replace("datasets/real_data/depths/", ""),
                                    "depth_path": depth_path.replace("datasets/real_data/depths/", "")}]
                except Exception as e:
                    print(f"path {curr_path}", e)


    recursive(start_path)
    create_folder_if_it_doesnt_exist(f'{base_path}/KITTI/annotations/')
    with open(f'{base_path}/annotations/train_annotations_onlyvideos.json', 'w') as fp:
        json.dump(annotation, fp)
