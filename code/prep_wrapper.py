import os
import argparse
import warnings
from osgeo import gdal

from lib import utils

def parse_call():
    """
    Function parses the CLI arguments and sends
    the appropriate variables to the image preprocessing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("src_ds",
                        help='''filename of source image to be processed''')
    parser.add_argument("dst_ds",
                        help='''filename of output image''')

    parser.add_argument("--training_label", type=str, default=None,
                        help="name of training classification list")
    parser.add_argument("-o", "--output_dir", type=str, default="default",
                        help="directory to place output results.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display text information and progress")
    parser.add_argument("-c", "--stretch",
                        type=str,
                        choices=["hist", "pansh", "toa_corr", "none"],
                        default='hist',
                        help='''Apply image correction/stretch to input: \n
                                   hist: Histogram stretch \n
                                   pansh: Orthorectify / Pansharpen for MS WV images \n
                                   none: No correction''')


def preprocess_image(src_file, dst_dir, dst_dtype=False,
                     adaptive_equalization=False, global_equalization=False,
                     white_balance=False):

    if adaptive_equalization and global_equalization:
        warnings.warn("Only one of adaptive and global equalization can be used. Defaulting to global.")
        adaptive_equalization = False

    # Filename suffixes to denote what processing was applied
    dst_file = generate_dst_fname(src_file, dst_dir, adaptive_equalization, global_equalization, white_balance)

    # Open the source dataset
    src_ds = gdal.Open(src_file, gdal.GA_ReadOnly)

    # If no dst_dtype was given, infer the type from the source file
    if dst_dtype is False:
        dst_dtype = src_ds.GetRasterBand(1).DataType

    # Create a blank copy of the source dataset
    dst_file = utils.blank_ds_from_source(src_ds, dst_file, dst_dtype, copy_bands=True, return_ds=False)
    src_ds = None

    if adaptive_equalization:
        apply_adaptive()


def apply_adaptive(src_file, dst_file):

    src_ds = gdal.Open(src_file, gdal.GA_ReadOnly)

    


def generate_dst_fname(src_file, dst_dir, adaptive_equalization, global_equalization, white_balance):

    file_suffix = ''
    if adaptive_equalization:
        file_suffix += 'ae'
    if global_equalization:
        file_suffix += 'ge'
    if white_balance:
        file_suffix += 'wb'

    # Compose the destination file by appending the suffix to the src file
    src_fname, src_ext = os.path.splitext(os.path.split(src_file)[1])
    dst_fname = os.path.join(dst_dir, src_fname + file_suffix + src_ext)

    return dst_fname


if __name__ == '__main__':
    parse_call()

