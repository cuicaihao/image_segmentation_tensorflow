# -*- coding: utf-8 -*-

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from splitraster import io


@click.command()
@click.argument('input_filepath', default='./data/raw/', type=click.Path(exists=True))
@click.argument('output_filepath', default='./data/processed/', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
    # print(ADMIN_EMAIL)
    # print(f"{input_filepath=}")
    # print(f"{output_filepath=}")

    input_image_path = input_filepath + "RGB.png"
    gt_image_path = input_filepath + "GT.png"

    save_path_rgb = output_filepath + "RGB"
    # crop_size = int(os.getenv("crop_size"))
    # repetition_rate = float(os.getenv("repetition_rate"))
    # overwrite = os.getenv("overwrite")
    crop_size = 256
    repetition_rate = 0.5
    overwrite = True

    print(overwrite)

    n = io.split_image(input_image_path, save_path_rgb, crop_size,
                       repetition_rate=repetition_rate, overwrite=overwrite)
    print(f"{n} tiles sample of {input_image_path} are added at {save_path_rgb}")

    save_path_gt = output_filepath + "GT"
    n = io.split_image(gt_image_path, save_path_gt, crop_size,
                       repetition_rate=repetition_rate, overwrite=overwrite)
    print(f"{n} tiles sample of {gt_image_path} are added at {save_path_gt}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
