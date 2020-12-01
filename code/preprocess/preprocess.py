# Code for preprocessing imagery for analysis by the OSSP image processing
# algorithm. Includes methods to split imagery into more manageable sections,
# and methods for histogram stretching to scale input images to the full 0,255
# range.
# Nicholas Wright
# 11/30/17

import os
import datetime
import subprocess
import numpy as np
import matplotlib.image as mimg
from skimage.measure import block_reduce
import rescale_intensity


def rescale_band(band, bottom, top):
    """
    Rescale and image data from range [bottom,top] to uint8 ([0,255])
    """
    imin, imax = (bottom, top)
    omin, omax = (1, 255)

    # Rescale intensity takes a uint16 dtype input
    band = band.astype(np.uint16)

    return rescale_intensity.rescale_intensity(band, imin, imax, omin, omax)


def white_balance(band, reference, omax):

    return rescale_intensity.white_balance(band, reference, omax)


def run_pgc_pansharpen(script_path, input_filepath, output_dir):

    base_cmd = os.path.join(script_path, 'pgc_pansharpen.py')

    cmd = 'python {} --epsg 3413 -c rf -t Byte --resample cubic {} {}'.format(
        base_cmd,
        input_filepath,
        output_dir)

    # Spawn a subprocess to execute the above command
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    proc.wait()

    # Copying PGC Naming convention, written to match above command
    basename = os.path.splitext(os.path.split(input_filepath)[-1])[0]
    pansh_filename = "{}_{}{}{}_pansh.tif".format(basename, 'u08', 'rf', '3413')

    return pansh_filename


def find_blocksize(x_dim, y_dim, desired_size):
    """
    Finds the appropriate block size for an input image of a given dimensions.
    Method returns the first factor of the input dimension that is greater than
        the desired size.
    """
    block_size_x = desired_size
    block_size_y = desired_size

    # Ensure that chosen block size divides into the image dimension with a remainder that is
    #   at least half a standard block in width.
    while (x_dim % block_size_x) <= (block_size_x / 2):
        block_size_x += 256
        # Make sure the blocks don't get too big.
        if block_size_x >= x_dim:
            block_size_x = x_dim
            break

    while (y_dim % block_size_y) <= (block_size_y / 2):
        block_size_y += 256
        if block_size_y >= y_dim:
            block_size_y = y_dim
            break

    return block_size_x, block_size_y


def calc_q_score(image):
    """
    Calculates a quality score of an input image by determining the number of
    high frequency peaks in the fourier transformed image relative to the
    image size.
    QA Score < 0.025          poor
    0.25 < QA Score < 0.035   medium
    QA Score > 0.035          fine

    """
    # Calculate the 2D fourier transform of the image
    im_fft = np.fft.fft2(image)
    # Find the maximum frequency peak in the fft image
    max_freq = np.amax(np.abs(im_fft))
    # Set a threshold that is a fraction of the max peak
    #  (Fraction determined empirically)
    thresh = max_freq / 100000
    # Determine the number of pixels above the threshold
    th = np.sum([im_fft>thresh])
    # QA Score is the percent of the pixels that are greater than the threshold
    qa_score = float(th) / np.size(image)

    return qa_score


def parse_metadata(metadata):
    """
    Parse image metadata information to find date.
    If image date cannot be found, return mean date of melt season (June1).
    This is likely to have to least impact on decision tree outcomes, as there
        will be less bias towards no melt vs lots of melt, but this should be tested.
        If 0 were to be used, then the decision tree would see that as very early
        season, since date is a numeric feature, and not a categorical one.
    """
    # Use june 1 as the default date if metadata read is unsuccessful
    default = True
    yyyy = 2014
    mm = 6
    dd = 1

    try:
        header_date = metadata['EXIF_DateTime']
    except KeyError:
        pass
    else:
        image_date = header_date[5:7] + header_date[8:10]
        yyyy = 2014
        mm = image_date[:2]
        dd = image_date[2:]
        default = False

    try:
        # image_date = metadata['NITF_STDIDC_ACQUISITION_DATE'][4:8]
        image_date = metadata['NITF_IDATIM'][0:8]
    except KeyError:
        pass
    else:
        yyyy = image_date[0:4]
        mm = image_date[4:6]
        dd = image_date[6:]
        default = False

    # Convert the date to julian day format (number of days since Jan 1)
    d = datetime.date(int(yyyy), int(mm), int(dd))
    doy = d.toordinal() - datetime.date(d.year, 1, 1).toordinal() + 1

    return doy, default


def histogram_threshold(gdal_dataset, src_dtype):
    # Set the percentile thresholds at a temporary value until finding the
    #   appropriate ones considering all bands. These numbers are chosen to
    #   always get reset on first loop (for bitdepth <= uint16)
    lower = 2048
    upper = -1

    # Determine the number of bands in the dataset
    band_count = gdal_dataset.RasterCount
    # White balance reference points
    wb_reference = [0 for _ in range(band_count)]
    bp_reference = [0 for _ in range(band_count)]
    # Determine the input datatype

    if src_dtype > 8:
        max_bit = 2047
        upper_limit = 0.25
    else:
        max_bit = 255
        upper_limit = 0.8

    total_peaks = 0

    # First for loop finds the threshold based on all bands
    for b in range(1, band_count + 1):

        # Read the band information from the gdal dataset
        band = gdal_dataset.GetRasterBand(b)

        # Find the min and max image values
        bmin, bmax = band.ComputeRasterMinMax()

        # Determine the histogram using gdal
        nbins = int(bmax - bmin)
        hist = band.GetHistogram(bmin, bmax, nbins, approx_ok=0)
        bin_centers = range(int(bmin), int(bmax))
        bin_centers = np.array(bin_centers)

        # Remove the image data from memory for now
        band = None

        # Find the strongest (3) peaks in the band histogram
        peaks = find_peaks(hist, bin_centers)
        print(peaks)
        peaks = trim_peaks(hist, bin_centers, peaks)
        print(peaks)
        # Tally the total number of peaks found across all bands
        total_peaks += len(peaks)
        # Find the high and low threshold for rescaling image intensity
        lower_b, upper_b, auto_wb, auto_bpr = find_threshold(hist, bin_centers,
                                                             peaks, src_dtype)
        wb_reference[b-1] = auto_wb
        bp_reference[b-1] = auto_bpr
        # For sRGB we want to scale each band by the min and max of all
        #   bands. Check thresholds found for this band against any that
        #   have been previously found, and adjust if necessary.
        if lower_b < lower:
            lower = lower_b
        if upper_b > upper:
            upper = upper_b

    # If there is only a single peak per band, we need an upper limit. The upper
    #   limit is to prevent open water only images from being stretched.
    if total_peaks <= band_count:
        max_range = int(max_bit * upper_limit)
        if upper < max_range:
            upper = max_range

    return lower, upper, wb_reference, bp_reference


def find_peaks(hist, bin_centers, width=None):
    """
    Finds peaks in histogram.
    :param hist: Nx1 array of histogram counts
    :param bin_centers: Nx1 array of the value for each bin
    :param width: acceptable width of a peak, in units of bins
    """

    # Roughly define the smallest acceptable size of a peak based on the number of pixels
    # in the largest bin.
    min_count = int(np.median(hist) * 0.2)
    # min_count = int(np.sum(hist)*.004)

    if width is None:
        width = int(len(bin_centers)*.02)

    # First find all potential peaks in the histogram
    peaks = []

    # Check the lowest histogram bin
    # if hist[0] >= hist[1] and hist[0] >= hist[5]:
    #     if hist[-1] > min_count:
    #         peaks.append(bin_centers[0])

    # Minimum width of peak in # bins
    kernel = [n for n in range(-1*width, 0)] + [n for n in range(1, width+1)]
    # Check the middle bins
    for i in range(len(bin_centers)):
        # Skip empty bins or those with too few counts
        if hist[i] < min_count:
            continue

        # Compare the current bin to those within the kernel
        is_peak = True
        for k in kernel:
            w = i + k
            if w < 0 or w > len(bin_centers)-1:
                continue
            if hist[i] <= hist[w]:
                is_peak = False
                break

        if is_peak:
            peaks.append(bin_centers[i])

    return peaks


def trim_peaks(hist, bin_centers, peaks):
    """
    Trims list of detected peaks to only the strongest 3
        Criteria for each peak:
        Distance to the nearest neighboring peak is greater than one third the approx. dynamic range of the input image
        Has a minimum number of pixels in that peak, loosely based on image size
        Is greater than the directly adjacent bins, and the bins +/- 5 away
    """
    num_peaks = len(peaks)
    distance = 5  # Initial distance threshold
    # One third the 'dynamic range' (radius from peak)
    distance_threshold = int((peaks[-1] - peaks[0]) / 6)
    # Min threshold
    if distance_threshold <= 5:
        distance_threshold = 5
    # Looking for three main peaks corresponding to the main surface types: 
    #   open water, MP and snow/ice
    # But any peak that passes the criteria is fine.
    while distance <= distance_threshold:
        i = 0
        to_remove = []
        # Cycle through all of the peaks
        while i < num_peaks - 1:
            # Check the current peak against the adjacent one. If they are closer 
            #   than the threshold distance, delete the lower valued peak
            if peaks[i + 1] - peaks[i] < distance:
                if (hist[np.where(bin_centers == peaks[i])[0][0]]
                        < hist[np.where(bin_centers == peaks[i + 1])[0][0]]):
                    to_remove.append(peaks[i])
                else:
                    to_remove.append(peaks[i + 1])
                    # Because we don't need to check the next peak again:
                    i += 1
            i += 1

        # Remove all of the peaks that did not meet the criteria above
        for j in to_remove:
            peaks.remove(j)

        # Recalculate the number of peaks left, and increase the distance threshold
        num_peaks = len(peaks)
        distance += 5

    return peaks


def find_threshold(hist, bin_centers, peaks, src_dtype, top=0.15, bottom=0.5):
    """
    Finds the upper and lower threshold for histogram stretching.
    Using the indices of the highest and lowest peak (by intensity, not # of pixels), this searches for an upper
    threshold that is both greater than the highest peak and has fewer than 15% the number of pixels, and a lower
    threshold that is both less than the lowest peak and has fewer than 50% the number of pixels.
    10% and 50% picked empirically to give good results.
    """
    max_peak = np.where(bin_centers == peaks[-1])[0][0]  # Max intensity
    thresh_top = max_peak
    while hist[thresh_top] > hist[max_peak] * top:
        thresh_top += 2  # Upper limit is less sensitive, so step 2 at a time
        # In the case that the top peak is already at/near the max bit value, limit the top
        #   threshold to be the top bin of the histogram.
        if thresh_top >= len(hist)-1:
            thresh_top = len(hist)-1
            break

    min_peak = np.where(bin_centers == peaks[0])[0][0]  # Min intensity
    thresh_bot = min_peak
    while hist[thresh_bot] > hist[min_peak] * bottom:
        thresh_bot -= 1
        # Similar to above, limit the bottom threshold to the lowest histogram bin.
        if thresh_bot <= 0:
            thresh_bot = 0
            break

    # Convert the histogram bin index to an intensity value
    lower = bin_centers[thresh_bot]
    upper = bin_centers[thresh_top]

    # Save the upper value for the auto white balance function
    auto_wb = upper
    # Save the lower value for the black point reference
    auto_bpr = lower

    # Determine the width of the lower peak.
    lower_width = min_peak - thresh_bot
    dynamic_range = max_peak - min_peak

    # Limit the amount of stretch to a percentage of the total dynamic range 
    #   in the case that all three main surface types are not represented (fewer
    #   than 3 peaks)
    # 8 bit vs 11 bit (WorldView)
    # 256   or 2048
    # While WV images are 11bit, white ice tends to be ~600-800 intensity
    # Provide a floor to the amount of stretch allowed
    if src_dtype > 8:
        max_bit = 2047
    else:
        max_bit = 255

    # If the width of the lowest peak is less than 3% of the bit depth,
    #   then the lower peak is likely open water. 3% determined visually, but
    #   ocean has a much narrower peak than ponds or ice.
    if (float(lower_width)/max_bit >= 0.03) or (dynamic_range < max_bit / 3):
        min_range = int(max_bit * .08)
        if lower > min_range:
            lower = min_range

    return lower, upper, auto_wb, auto_bpr
