import numpy as np
from osgeo import gdal


def imsave_gdal(dst_filename, raster_data, fileformat="GTiff", dst_dtype=gdal.GDT_Byte,
                save_opts=["TILED=YES", "COMPRESS=LZW"]):
    """Write raster data to disk using gdal drivers

    Parameters
    ----------
    dst_filename : str
        filename of saved image
    raster_data : 2d or 3d array
        image data to save
    fileformat : str, optional
        gdal driver name, by default "GTiff"
    dst_dtype : dtype, optional
        data type of output file, by default gdal.GDT_Byte
    save_opts : list, optional
        gdal creation options, by default ["TILED=YES", "COMPRESS=LZW"]
    """
    if raster_data.ndim == 3:
        x_dim, y_dim, n_bands = np.shape(raster_data)
    else:
        x_dim, y_dim = np.shape(raster_data)
        n_bands = 1

    driver = gdal.GetDriverByName(fileformat)
    dst_ds = driver.Create(dst_filename, xsize=x_dim, ysize=y_dim,
                           bands=n_bands, eType=dst_dtype, options=save_opts)

    for b in range(1, n_bands+1):
        dst_ds.GetRasterBand(b).WriteArray(raster_data[:,:,b-1])
        dst_ds.FlushCache()

    dst_ds = None