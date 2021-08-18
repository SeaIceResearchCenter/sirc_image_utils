import numpy as np
from osgeo import gdal


def blank_ds_from_source(src_ds, dst_filename, dst_dtype=gdal.GDT_Byte, copy_bands=False, rgb_only=False,
                         return_ds=True):
    """
    Copies the image parameters from the src_ds to a new file at dst_filename
    Optional arguments to alter the dst data type and number of bands

    :param src_ds: gdal dataset whose structure will be copied
    :param dst_filename: filename of new dataset
    :param dst_dtype: datatype of new dataset
    :param copy_bands: whether to create a dataset with 1 band (False) or nbands in src_ds (True)
    :param rgb_only: returns an output dataset with 3 bands, regardless of input
    :param return_ds: whether to return the open gdal dataset (True)
                           or write to disk and return the filename (False)
    :return: dst_ds: new gdal dataset
    """
    x_dim = src_ds.RasterXSize
    y_dim = src_ds.RasterYSize
    if copy_bands:
        if rgb_only:
            n_bands = 3
        else:
            n_bands = src_ds.RasterCount
    else:
        n_bands = 1

    # Create a blank output image dataset
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    dst_ds = driver.Create(dst_filename, xsize=x_dim, ysize=y_dim,
                           bands=n_bands, eType=dst_dtype, options=["TILED=YES", "COMPRESS=LZW"])

    # Transfer the metadata from input image
    dst_ds.SetMetadata(src_ds.GetMetadata())
    # Transfer the input projection and geotransform if they are different than the default
    if src_ds.GetGeoTransform() != (0, 1, 0, 0, 0, 1):
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())  # sets same geotransform as input
    if src_ds.GetProjection() != '':
        dst_ds.SetProjection(src_ds.GetProjection())  # sets same projection as input

    # Either returns the gdal dataset or closes the dataset and returns the filename
    if return_ds:
        return dst_ds
    else:
        dst_ds = None
        return dst_filename


def reshape_yxb2bxy(img_data, dtype=np.uint8):
    y, x, bands = np.shape(img_data)

    new_img = np.zeros((bands, y, x), dtype=dtype)
    for b in range(bands):
        new_img[b, :, :] = img_data[:, :, b]

    return new_img


def reshape_bxy2yxb(img_data, dtype=np.uint8):
    bands, x, y = np.shape(img_data)

    new_img = np.zeros((x, y, bands), dtype=dtype)
    for b in range(bands):
        new_img[:, :, b] = img_data[b, :, :]

    return new_img


# Combines multiple bands (RBG) into one 3D array
# Adapted from:  http://gis.stackexchange.com/questions/120951/merging-multiple-16-bit-image-bands-to-create-a-true-color-tiff
# Useful band combinations: http://c-agg.org/cm_vault/files/docs/WorldView_band_combs__2_.pdf
def create_composite(band_list, dtype=np.uint8):
    img_dim = np.shape(band_list[0])
    num_bands = len(band_list)
    img = np.zeros((img_dim[0], img_dim[1], num_bands), dtype=dtype)
    for i in range(num_bands):
        img[:, :, i] = band_list[i]
    
    return img