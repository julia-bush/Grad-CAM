from os import walk, remove
from argparse import ArgumentParser
import tensorflow as tf


def try_load_imgages(args):

    _, dir_names, _ = next(walk(args.sort_dir))
    for dir_name in dir_names:
        print(dir_name)
        _, _, img_filenames = next(walk(f"{args.sort_dir}/{dir_name}"))
        for img_filename in img_filenames:
            path = f"{args.sort_dir}/{dir_name}/{img_filename}"
            target_size = (224, 224, 3)[:-1]
            try:
                tf.keras.preprocessing.image.load_img(
                    path, grayscale=False, color_mode='rgb', target_size=target_size,
                    interpolation='nearest'
                )
            except:
                with open(f"{args.log_dir}/failed_img_downloads.txt", "a") as f:
                    f.write(str(img_filename) + "\n")
                print(f"{img_filename} failed to load. Deleted and logged.")
                remove(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--log_dir", default="C:/Users/j/Documents/PhD/Data/scrapy/logs/preprocessing", type=str)
    # parser.add_argument("--sort_dir", default="C:/Users/j/Documents/PhD/Data/scrapy/sorted", type=str)
    parser.add_argument("--log_dir", default="/app/logs", type=str)  # mounted volume in docker container
    parser.add_argument("--sort_dir", default="/app/data/HE_defects", type=str)  # mounted volume in docker container
    args = parser.parse_args()
    try_load_imgages(args=args)
