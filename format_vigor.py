import os
import shutil
import util
import map_builder
import numpy as np
from PIL import Image
from distutils.dir_util import copy_tree

def format_querie_name(src_image_name : str) -> str:
    lat, long = src_image_name.split(',')[1:3]
    return util.get_dst_image_name(lat, long)

def format_database_name(src_image_name : str) -> str:
    lat, long = src_image_name.split('_')[1:3]
    long = os.path.splitext(long)[0]
    return util.get_dst_image_name(lat, long)

def format_image_name(src_image_name : str, data_type : str) -> str:
    if data_type == "database":
        return format_database_name(src_image_name)
    else:
        return format_querie_name(src_image_name)

def change_image_to_jpg(image_path : str) -> None:
    image_png = Image.open(image_path, mode='r', formats=None).convert('RGB')
    image_png.save(image_path, format="JPEG")

def find_all_images(path : str) -> 'tuple(str, str)':
    for src_name in os.listdir(path):
        dataset_folder = os.path.join(path, src_name)
        for src_dataset_name in os.listdir(dataset_folder):
            image_folder = os.path.join(dataset_folder, src_dataset_name)
            for src_image_name in os.listdir(image_folder):
                yield image_folder, src_image_name

def format_images(src_folder : str) -> None:
    src_folder = os.path.join(src_folder, "images")
    os.makedirs(src_folder, exist_ok=True)
    for image_folder, src_image_name in find_all_images(src_folder):
        if src_image_name[0] == "@":
            continue
        src_path = os.path.join(image_folder, src_image_name)
        if os.path.splitext(src_image_name)[1] == ".png": 
            change_image_to_jpg(src_path)
        dst_image_name = format_image_name(src_image_name, os.path.basename(image_folder))
        dst_path = os.path.join(image_folder, dst_image_name)
        os.makedirs(image_folder, exist_ok=True)
        shutil.move(src_path, dst_path)

def generate_queries_positives(split_file : str) -> 'tuple[str, list]':
    for line in open(split_file):
        line = line.split(' ')
        yield line[0], [line[1], line[4], line[7], line[10]]

def split_dataset(src_folder : str, dst_folder : str, splits : dict) -> None:
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    split_file = os.path.join(os.path.dirname(src_folder), "splits", os.path.basename(src_folder), "pano_label_balanced.txt")
    for querie, positives in generate_queries_positives(split_file):
        choice = np.random.choice(len(splits.keys()), 1, p=list(splits.values()))[0]
        choice = list(splits.keys())[choice]
        src_path = os.path.join(src_folder, "queries", querie)
        dst_path = os.path.join(dst_folder, "images", choice, "queries", querie)
        shutil.copy(src_path, dst_path)
        for positive in positives:
            src_path = os.path.join(src_folder, "database", positive)
            dst_path = os.path.join(dst_folder, "images", choice, "database", positive)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

def split_dataset_small(src_folder : str, dst_folder : str) -> None:
    os.makedirs(os.path.join(dst_folder, "images", "train", "database"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "images", "train", "queries"), exist_ok=True)
    split_file = os.path.join(os.path.dirname(src_folder), "splits", os.path.basename(src_folder), "pano_label_balanced.txt")
    for i, (querie, positives) in enumerate(generate_queries_positives(split_file)):
        if i == 500:
            break
        src_path = os.path.join(src_folder, "queries", querie)
        dst_path = os.path.join(dst_folder, "images", "train", "queries", querie)
        shutil.copy(src_path, dst_path)
        for positive in positives:
            src_path = os.path.join(src_folder, "database", positive)
            dst_path = os.path.join(dst_folder, "images", "train", "database", positive)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
    copy_tree(os.path.join(dst_folder, "images", "train"), os.path.join(dst_folder, "images", "val"))
    copy_tree(os.path.join(dst_folder, "images", "train"), os.path.join(dst_folder, "images", "test"))

def split_database_only(src_folder : str, dst_folder : str, splits : dict) -> None:
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    src_path = os.path.join(src_folder, "database")
    for image in os.listdir(src_path):
        choice = np.random.choice(len(splits.keys()), 1, p=list(splits.values()))[0]
        choice = list(splits.keys())[choice]
        if np.random.choice(2, 1)[0]:
            dst_path = os.path.join(dst_folder, "images", choice, "database", image)
        else:
            dst_path = os.path.join(dst_folder, "images", choice, "queries", image)
        shutil.copy(os.path.join(src_path, image), dst_path)

def split_database_only2(src_folder : str, dst_folder : str, splits : dict) -> None:
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    src_path = os.path.join(src_folder, "database")
    min_lat = 90
    max_lat = -90
    for image in os.listdir(src_path):
        lat = float(image.split('_')[1])
        if lat < min_lat:
            min_lat = lat
        if lat > max_lat:
            max_lat = lat
    train_lat = min_lat + (max_lat - min_lat) * splits['train']
    val_lat = min_lat + (max_lat - min_lat)  * splits['val']
    for image in os.listdir(src_path):
        lat = float(image.split('_')[1])
        if np.random.choice(9, 1)[0] > 0:
            if lat < train_lat:
                dst_path = os.path.join(dst_folder, "images", "train", "database", image)
                shutil.copy(os.path.join(src_path, image), dst_path)
            if lat < val_lat:
                dst_path = os.path.join(dst_folder, "images", "val", "database", image)
                shutil.copy(os.path.join(src_path, image), dst_path)
            if lat >= train_lat:
                dst_path = os.path.join(dst_folder, "images", "test", "database", image)
                shutil.copy(os.path.join(src_path, image), dst_path)
        else:
            if lat < train_lat:
                dst_path = os.path.join(dst_folder, "images", "train", "queries", image)
                shutil.copy(os.path.join(src_path, image), dst_path)
            if lat < val_lat:
                dst_path = os.path.join(dst_folder, "images", "val", "queries", image)
                shutil.copy(os.path.join(src_path, image), dst_path)
            if lat >= train_lat:
                dst_path = os.path.join(dst_folder, "images", "test", "queries", image)
                shutil.copy(os.path.join(src_path, image), dst_path)

def split_queries_only(src_folder : str, dst_folder : str, splits : dict) -> None:
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    src_path = os.path.join(src_folder, "queries")
    for image in os.listdir(src_path):
        choice = np.random.choice(len(splits.keys()), 1, p=list(splits.values()))[0]
        choice = list(splits.keys())[choice]
        if np.random.choice(2, 1)[0]:
            dst_path = os.path.join(dst_folder, "images", choice, "database", image)
        else:
            dst_path = os.path.join(dst_folder, "images", choice, "queries", image)
        shutil.copy(os.path.join(src_path, image), dst_path)

def format_queries(src_folder : str) -> None:
    src_folder = os.path.join(src_folder, "images")
    os.makedirs(src_folder, exist_ok=True)
    for image_folder, src_image_name in find_all_images(src_folder):
        if src_image_name[0] == "@":
            continue
        src_path = os.path.join(image_folder, src_image_name)
        if os.path.splitext(src_image_name)[1] == ".png": 
            change_image_to_jpg(src_path)
        dst_image_name = format_image_name(src_image_name, "queries")
        dst_path = os.path.join(image_folder, dst_image_name)
        os.makedirs(image_folder, exist_ok=True)
        shutil.move(src_path, dst_path)

def format_database(src_folder : str) -> None:
    src_folder = os.path.join(src_folder, "images")
    os.makedirs(src_folder, exist_ok=True)
    for image_folder, src_image_name in find_all_images(src_folder):
        if src_image_name[0] == "@":
            continue
        src_path = os.path.join(image_folder, src_image_name)
        if os.path.splitext(src_image_name)[1] == ".png": 
            change_image_to_jpg(src_path)
        dst_image_name = format_image_name(src_image_name, "database")
        dst_path = os.path.join(image_folder, dst_image_name)
        os.makedirs(image_folder, exist_ok=True)
        shutil.move(src_path, dst_path)

def preapare_dataset(dataset_name : str, src_folder : str) -> None:
    dataset_folder = os.path.join(os.curdir, "datasets_vg", "datasets", dataset_name)
    # os.makedirs(dataset_folder, exist_ok=True)
    # os.makedirs(src_folder, exist_ok=True)
    # #split_dataset(src_folder, dataset_folder, {"train": 0.7, "val": 0.15, "test": 0.15})
    # #split_dataset_small(src_folder, dataset_folder)
    split_database_only2(src_folder, dataset_folder, {"train": 0.7, "val": 0.15, "test": 0.15})
    format_database(dataset_folder)
    map_builder.build_map_from_dataset(dataset_folder)
    #shutil.rmtree(raw_data_folder)

if __name__ == "__main__":
    preapare_dataset("VIGOR_DATABASE2", "VIGOR/Chicago")
