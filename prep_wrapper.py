import os
import argparse
import warnings

import numpy as np
from osgeo import gdal
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage import color

from lib import utils
from sirc_image_utils import preprocess as pp

import matplotlib
import matplotlib.pyplot as plt
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank


def parse_call():
    """
    Function parses the CLI arguments and sends
    the appropriate variables to the image preprocessing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file",
                        help='''filename of source image to be processed''')
    parser.add_argument("dst_dir",
                        help='''filename of output image''')
    parser.add_argument("-a", "--adaptive", action="store_true",
                        help="Apply adaptive histogram equalization")
    parser.add_argument("-w", "--wbalance", action="store_true",
                        help="Apply automatic white balance")

    # Parse Arguments
    args = parser.parse_args()

    # System filepath that contains the directories or files for batch processing
    src_file = args.src_file
    if not os.path.isfile(src_file):
        raise IOError('Invalid input')
    dst_dir = args.dst_dir
    ae = args.adaptive
    wb = args.wbalance

    if src_file != "all":
        preprocess_image(src_file, dst_dir, adaptive_equalization=ae, white_balance=wb)
        quit()

    base_dir = '/media/ncwright/sequoia/Lagrangian_Tracking/2020/imagery/'

    acceptable_cf = ['CF1', 'CF2']

    buoy_id = '300534060314060'
    source_dir = os.path.join(base_dir, buoy_id)
    # Build a list of all the good images and the associated metadata file
    img_list = get_image_list(source_dir, acceptable_cf=acceptable_cf)

    for img, md in img_list:
        print(img)
        dst_dir = os.path.split(img)[0]
        print(dst_dir)
        preprocess_image(img, dst_dir, adaptive_equalization=ae, white_balance=wb)
        # quit()


# Temp just to process all of one buoys files:
def get_image_list(directory, acceptable_cf=None):
    """
    Given a directory, returns a list of (image, metadata) matched pairs
    """
    im_file = None
    md_file = None
    im_list = []
    if acceptable_cf is None:
        acceptable_cf = ['CF1']

    im_files = []
    # This is a very error prone way of searching the directory...
    for dir_ in os.listdir(directory):
        dir_ = os.path.join(directory, dir_)
        if dir_.split('_')[-2] in acceptable_cf: # and path_.split('_')[-1].split('/')[0] == 'H1':
            for f in os.listdir(dir_):
                if os.path.splitext(f)[1] == '.json':
                    md_file = os.path.join(dir_, f)
            analytic_dir = os.path.join(dir_, 'analytic')
            for f in os.listdir(analytic_dir):
                # Compare [-1] to 'analytic.tif' for raw file
                if f.split('_')[-1][0] == 'p' and os.path.splitext(f)[-1] == '.tif':
                    im_files.append(os.path.join(analytic_dir, f))

            for im_file in im_files:
                if md_file is not None:
                    if md_file.split('_')[1] == im_file.split('_')[1]:
                        im_list.append((im_file, md_file))

            md_file = None
            im_files = []

    return im_list


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
    dst_file = utils.blank_ds_from_source(src_ds, dst_file, dst_dtype, copy_bands=True, rgb_only=True, return_ds=False)
    src_ds = None

    img_data = load_image(src_file)

    if adaptive_equalization:
        print("Applying adaptive equalization")
        img_data = apply_adaptive_3d(img_data)

    if white_balance:
        # Numpy uses the format (Y,X,B) for images, gdal uses (B,X,Y)
        img_data = utils.reshape_yxb2bxy(img_data)

        print("Calculating image statistics")
        lower, upper, wb_ref, bp_ref = pp.histogram_threshold(img_data, 8, gdal_dataset=False)
        print(lower, upper, wb_ref)
        print("Applying white balance")
        wb_ref = np.array(wb_ref, dtype=np.float)
        max_wb = np.amax(wb_ref)
        img_data = pp.white_balance(img_data, wb_ref, int((255+max_wb+max_wb)/3))
        # Put it back in np format
        img_data = utils.reshape_bxy2yxb(img_data)

    save_image(img_data, dst_file)


def load_image(src_file):
    print("Reading image data...")
    src_ds = gdal.Open(src_file, gdal.GA_ReadOnly)

    min_count, max_count = lookup_minmax(os.path.split(src_file))
    # lower, upper, wb_ref, bp_ref = pp.histogram_threshold(src_ds, 16, gdal_dataset=True, ignore_nir=True,
    #                                                       top=0.05, bottom=0.25)
    #
    # print(lower, upper)
    lower, upper = pp.cumulative_dist_thresh(src_ds, gdal_dataset=True, ignore_nir=True,
                                             min_count=0.02, max_count=.98)

    # print(lower, upper)
    # lower = [2115, 2942, 149]
    # upper = [6532, 5876, 3124]
    # print(lower, upper)
    # print(wb_ref)
    # print(bp_ref)
    # src_ds = None
    # quit()
    band_list = []
    band_order = [3, 2, 1]
    for b in band_order:
        band = src_ds.GetRasterBand(b).ReadAsArray()
        # cdf, centers = exposure.cumulative_distribution(band)
        # plt.figure()
        # plt.plot(centers, cdf, 'r')
        # plt.show()
        # print(np.amax(band))
        # band = img_as_ubyte(band)
        # Maybe change the min/max scaling to some %count or stdev
        band = pp.rescale_band(band, lower[b-1]+1, upper[b-1])
        # print(np.amax(band))
        # plt.figure()
        # plt.imshow(band, cmap='Greys_r')
        # plt.show()
        band_list.append(band)
    src_ds = None

    img = utils.create_composite(band_list)

    return img


def save_image(img, dst_file):

    dst_ds = gdal.Open(dst_file, gdal.GA_Update)
    for b in range(1, 4):
        dst_ds.GetRasterBand(b).WriteArray(img[:, :, b-1])

    dst_ds.FlushCache()
    dst_ds = None

    return dst_file


def apply_adaptive_3d(img):

    # Convert image from rgb to hsv
    hsv_img = color.rgb2hsv(img)

    img_v = hsv_img[:, :, 2]

    # Apply the adaptive histogram to the value channel
    hsv_img[:, :, 2] = exposure.equalize_adapthist(hsv_img[:, :, 2], clip_limit=0.01, kernel_size=400)

    # Convert back to rgb space
    final_img = color.hsv2rgb(hsv_img)

    final_img = img_as_ubyte(final_img)

    # Display results
    # fig = plt.figure(figsize=(8, 5))
    # axes = np.zeros((2, 3), dtype=np.object)
    # axes[0, 0] = plt.subplot(2, 3, 1)
    # axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    # axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    # axes[1, 0] = plt.subplot(2, 3, 4)
    # axes[1, 1] = plt.subplot(2, 3, 5)
    # axes[1, 2] = plt.subplot(2, 3, 6)
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    # ax_img.set_title('Original image')
    # ax_hist.set_ylabel('Number of pixels')
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(hsv_img[:, :, 2], axes[:, 1])
    # ax_img.set_title('Value')
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(final_img, axes[:, 2])
    # ax_img.set_title('Final Image')
    # ax_cdf.set_ylabel('Fraction of total intensity')
    #
    # # prevent overlap of y-axis labels
    # fig.tight_layout()
    # plt.show()

    return final_img


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf


def generate_dst_fname(src_file, dst_dir, adaptive_equalization, global_equalization, white_balance):

    file_suffix = '_P'
    if adaptive_equalization:
        file_suffix += '_ae'
    if global_equalization:
        file_suffix += '_ge'
    if white_balance:
        file_suffix += '_wb'

    # Compose the destination file by appending the suffix to the src file
    src_fname, src_ext = os.path.splitext(os.path.split(src_file)[1])
    dst_fname = os.path.join(dst_dir, src_fname + file_suffix + src_ext)

    return dst_fname

# Custom stretch params
def lookup_minmax(src_file):

    try:
        min_, max_ = {'20200530_002534_ssc6_u0001_analytic_p02.tif': (.02, .98),
                      }[src_file]
    except KeyError:
        return 0.02, 0.98
    return


if __name__ == '__main__':
    parse_call()

