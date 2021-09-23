import numpy as np


def load_histogram(src_ds, b, gdal_dataset):
    """Loads the histogram from a single band of the given dataset

    Parameters
    ----------
    src_ds : gdaldataset or array_like
        Image dataset on which to compute histogram.
        Either a gdal dataset (return of gdal.Open()) or a raster array
    b : int
        band number of src_ds, 1 indexed
    gdal_dataset : bool
        whether src_ds is a gdal dataset (True) or a raster array (False)

    Returns
    -------
    hist : numpy array of int
        bin counts for each intensity value of band in src_ds
    bin_centers : numpy array of int
        intensity value of each bin in hist
        
    """

    if gdal_dataset:
        # Read the band information from the gdal dataset
        band = src_ds.GetRasterBand(b)

        # Find the min and max image values
        bmin, bmax = band.ComputeRasterMinMax()

        # Determine the histogram using gdal
        nbins = int(bmax - bmin + 1)
        hist = band.GetHistogram(bmin, bmax, nbins, approx_ok=0)

        # Remove the image data from memory for now
        band = None

    else:
        # Read the current band from the image
        band = src_ds[b - 1, :, :]

        # Find the min and max image values
        bmin = np.amin(band)
        bmax = np.amax(band)

        # Determine the histogram using numpy
        nbins = int(bmax - bmin + 1)
        hist, bin_edges = np.histogram(band, bins=nbins, range=(bmin, bmax))

        # Remove the image data from memory for now
        band = None

    ####### This didnt have +1 when testing with gdal_dataset = False, need to retest that case
    bin_centers = range(int(bmin), int(bmax)+1)
    bin_centers = np.array(bin_centers)

    return hist, bin_centers


def cumulative_dist_thresh(src_ds, gdal_dataset=True, ignore_nir=False, min_count=0.01, max_count=0.99):
    """Calculates upper and lower threshold using the cumulative distribution method

    Parameters
    ----------
    src_ds : gdaldataset or array_like
        raster data on which to caclulate threshold values
    gdal_dataset : bool, optional
        whether src_ds is a gdal dataset (returned by gdal.Open()), by default True
    ignore_nir : bool, optional
        ignore the nir band (band 4, typically), by default False
    min_count : float, optional
        lower threshold value, by default 0.01
    max_count : float, optional
        upper threshold value, by default 0.99

    Returns
    -------
    lower : int
        intenisty threshold value
    upper : int
        intensity threshold value
    """

    # Determine the number of bands in the dataset
    if gdal_dataset:
        band_count = src_ds.RasterCount
    else:
        band_count = np.shape(src_ds)[0]

    if ignore_nir:
        band_count -= 1

    lower = [0 for _ in range(band_count)]
    upper = [0 for _ in range(band_count)]
    import matplotlib.pyplot as plt
    for b in range(1, band_count + 1):
        hist, bin_centers = load_histogram(src_ds, b, gdal_dataset)

        cdf = np.cumsum(hist)
        cdf = (cdf / np.amax(cdf))

        # Set default values
        lower_b = 0
        upper_b = len(cdf)

        lower_first = False
        upper_first = False
        for i in range(len(cdf)):
            if cdf[i] > min_count and lower_first is False:
                lower_b = bin_centers[i]
                lower_first = True
            if cdf[i] > max_count and upper_first is False:
                upper_b = bin_centers[i]
                upper_first = True

        lower[b-1] = lower_b
        upper[b-1] = upper_b
        # print(lower_b, upper_b)
        # plt.figure()
        # plt.plot(bin_centers, cdf)
        # plt.figure()
        # plt.plot(bin_centers, hist)
        # plt.show()

    return lower, upper