import os
import shutil
import util
import map_builder
import numpy as np
from tqdm import tqdm
from PIL import Image

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

def find_all_archives(path : str, archive_type : str) -> 'tuple(str, str)':
    for src_name in os.listdir(path):
        zip_folder = os.path.join(path, src_name)
        for zip_name in os.listdir(zip_folder):
            if os.path.splitext(zip_name)[1] == archive_type:
                yield zip_folder, zip_name

def generate_queries_positives(split_file : str) -> 'tuple[str, list]':
    for line in open(split_file):
        line = line.split(' ')
        yield line[0], [line[1], line[4], line[7], line[10]]

def lat_generator(src_folder : str, dataset : str) -> None:
    split = {"queries" : ",", "database" : "_"}
    for image in os.listdir(src_folder):
        yield float(image.split(split[dataset])[1])

def split_dataset_from_file(src_folder : str, dst_folder : str, splits : dict) -> None:
    split = {"queries" : ",", "database" : "_"}
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    split_file = os.path.join(os.path.dirname(src_folder), "splits", os.path.basename(src_folder), "pano_label_balanced.txt")
    min_lat = min(lat_generator(os.path.join(src_folder, "database"), "database"))
    max_lat = max(lat_generator(os.path.join(src_folder, "database"), "database"))
    train_lat = min_lat + (max_lat - min_lat) * splits['train']
    val_lat = min_lat + (max_lat - min_lat)  * splits['val']
    for (querie, positives) in tqdm(list(generate_queries_positives(split_file))):
        src_path = os.path.join(src_folder, "queries", querie)
        lat = float(src_path.split(split["queries"])[1])
        if lat < train_lat:
            dst_image_name = format_image_name(querie, "queries")
            dst_path = os.path.join(dst_folder, "images", "train", "queries", dst_image_name)
            shutil.copy(src_path, dst_path)
            if os.path.splitext(querie)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            for positive in positives:
                dst_image_name = format_image_name(positive, "database")
                src_path = os.path.join(src_folder, "database", positive)
                dst_path = os.path.join(dst_folder, "images", "train", "database", dst_image_name)
                shutil.copy(src_path, dst_path)
                if os.path.splitext(positive)[1] == ".png": 
                    change_image_to_jpg(dst_path)
        if lat >= train_lat:
            dst_image_name = format_image_name(querie, "queries")
            dst_path = os.path.join(dst_folder, "images", "test", "queries", dst_image_name)
            shutil.copy(src_path, dst_path)
            if os.path.splitext(querie)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            for positive in positives:
                dst_image_name = format_image_name(positive, "database")
                src_path = os.path.join(src_folder, "database", positive)
                dst_path = os.path.join(dst_folder, "images", "test", "database", dst_image_name)
                shutil.copy(src_path, dst_path)
                if os.path.splitext(positive)[1] == ".png": 
                    change_image_to_jpg(dst_path)
        if lat < val_lat:
            dst_image_name = format_image_name(querie, "queries")
            dst_path = os.path.join(dst_folder, "images", "val", "queries", dst_image_name)
            shutil.copy(src_path, dst_path)
            if os.path.splitext(querie)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            for positive in positives:
                dst_image_name = format_image_name(positive, "database")
                src_path = os.path.join(src_folder, "database", positive)
                dst_path = os.path.join(dst_folder, "images", "val", "database", dst_image_name)
                shutil.copy(src_path, dst_path)
                if os.path.splitext(positive)[1] == ".png": 
                    change_image_to_jpg(dst_path)

def split_dataset(src_folder : str, dst_folder : str, splits : dict) -> None:
    split = {"queries" : ",", "database" : "_"}
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    min_lat = min(lat_generator(os.path.join(src_folder, "database"), "database"))
    max_lat = max(lat_generator(os.path.join(src_folder, "database"), "database"))
    train_lat = min_lat + (max_lat - min_lat) * splits['train']
    val_lat = min_lat + (max_lat - min_lat)  * splits['val']
    print("Starting to copy images")
    for dataset in os.listdir(src_folder):
        if os.path.splitext(dataset)[1] == ".gz":
            continue
        dataset_folder = os.path.join(src_folder, dataset)
        for image in tqdm(os.listdir(dataset_folder)):
            dst_image_name = format_image_name(image, dataset)
            lat = float(image.split(split[dataset])[1])
            if lat < train_lat:
                dst_path = os.path.join(dst_folder, "images", "train", dataset, dst_image_name)
                shutil.copy(os.path.join(dataset_folder, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat >= train_lat:
                dst_path = os.path.join(dst_folder, "images", "test", dataset, dst_image_name)
                shutil.copy(os.path.join(dataset_folder, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat < val_lat:
                dst_path = os.path.join(dst_folder, "images", "val", dataset, dst_image_name)
                shutil.copy(os.path.join(dataset_folder, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)


def split_one_dataset_only(src_folder : str, dst_folder : str, dataset : str, splits : dict) -> None:
    split = {"queries" : ",", "database" : "_"}
    for dataset_type in splits:
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "database"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, "images", dataset_type, "queries"), exist_ok=True)
    src_path = os.path.join(src_folder, dataset)
    min_lat = min(lat_generator(src_path, dataset))
    max_lat = max(lat_generator(src_path, dataset))
    train_lat = min_lat + (max_lat - min_lat) * splits['train']
    val_lat = min_lat + (max_lat - min_lat)  * splits['val']
    print("Starting to copy images")
    for image in tqdm(os.listdir(src_path)):
        dst_image_name = format_image_name(image, dataset)
        lat = float(image.split(split[dataset])[1])
        if np.random.choice(5, 1)[0] > 0:
            if lat < train_lat:
                dst_path = os.path.join(dst_folder, "images", "train", "database", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat >= train_lat:
                dst_path = os.path.join(dst_folder, "images", "test", "database", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat < val_lat:
                dst_path = os.path.join(dst_folder, "images", "val", "database", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
        else: 
            if lat < train_lat:
                dst_path = os.path.join(dst_folder, "images", "train", "queries", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat >= train_lat:
                dst_path = os.path.join(dst_folder, "images", "test", "queries", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)
            if lat < val_lat:
                dst_path = os.path.join(dst_folder, "images", "val", "queries", dst_image_name)
                shutil.copy(os.path.join(src_path, image), dst_path)
                if os.path.splitext(image)[1] == ".png": 
                    change_image_to_jpg(dst_path)

def unpack_data(src_folder : str, dst_folder : str, delete_archives : bool = False) -> None:
    for zip_folder, zip_name in find_all_archives(src_folder, ".gz"):
        names = {"panorama.tar.gz" : "queries", "satellite.tar.gz" : "database"}
        if zip_name not in names:
            continue
        shutil.unpack_archive(os.path.join(zip_folder, zip_name), os.path.join(dst_folder, names[zip_name]))
        if delete_archives:
            os.remove(os.path.join(zip_folder, zip_name))

def preapare_dataset(src_folder : str, dataset_name : str) -> None:
    dataset_folder = os.path.join(os.curdir, "datasets_vg", "datasets", dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(src_folder, exist_ok=True)
    split_dataset(src_folder, dataset_folder, {"train": 0.7, "val": 0.15, "test": 0.15})
    map_builder.build_map_from_dataset(dataset_folder)

def preapare_one_dataset_only(src_folder : str, dataset_name : str, dataset) -> None:
    dataset_folder = os.path.join(os.curdir, "datasets_vg", "datasets", dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(src_folder, exist_ok=True)
    split_one_dataset_only(src_folder, dataset_folder, dataset, {"train": 0.7, "val": 0.15, "test": 0.15})
    map_builder.build_map_from_dataset(dataset_folder)

def preapare_dataset_from_file(src_folder : str, dataset_name : str) -> None:
    dataset_folder = os.path.join(os.curdir, "datasets_vg", "datasets", dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(src_folder, exist_ok=True)
    split_dataset_from_file(src_folder, dataset_folder, {"train": 0.7, "val": 0.15, "test": 0.15})
    map_builder.build_map_from_dataset(dataset_folder)

if __name__ == "__main__":
    # unpack_data(src_folder, src_folder)
    # preapare_one_dataset_only("VIGOR/Chicago", "VIGOR_QUERIES", "queries")
    # preapare_dataset("VIGOR/Chicago", "VIGOR")
    preapare_dataset_from_file("VIGOR/Chicago", "VIGOR_FILE")