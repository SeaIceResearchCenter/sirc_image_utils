# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
cimport cython
import numpy as np
from ctypes import *


def white_balance(src_ds, reference, double omax):
    '''
    src_ds: input image to balance (ndim must == 3)
    reference: array of length equal to src_ds dim 0, image will be scaled by this reference
    '''
    cdef int x, y, b
    cdef int x_dim, y_dim, num_bands
    cdef float val
    cdef unsigned short new_val_short
    cdef unsigned char new_val

    cdef unsigned char [:, :, :] src_view = src_ds
    dst_ds = np.empty_like(src_ds)
    cdef unsigned char [:, :, :] dst_view = dst_ds
    cdef double [:] ref_view = reference

    num_bands, x_dim, y_dim = np.shape(src_ds)

    # Check that the user provided the correct number of reference points
    if np.shape(reference)[0] != num_bands:
        return src_ds

    for y in range(y_dim):
        for x in range(x_dim):
            for b in range(num_bands):
                val = src_view[b, x, y]
                if val == 0:
                    new_val = 0
                else:
                    new_val_short = int((omax / ref_view[b]) * val)
                    if new_val_short < 1:
                        new_val = 1
                    elif new_val_short > 255:
                        new_val = 255
                    else:
                        new_val = new_val_short
                dst_view[b, x, y] = new_val

    return np.copy(dst_view)


def rescale_intensity(src_ds, int imin, int imax, int omin, int omax):

    # Check raster structure for panchromatic images:
    if src_ds.ndim == 3:
        return _rescale_intensity_3d(src_ds, imin, imax, omin, omax)
    else:
        return _rescale_intensity_2d(src_ds, imin, imax, omin, omax)


def _rescale_intensity_3d(src_ds, int imin, int imax, int omin, int omax):
    '''
    Rescales the input image intensity values.
    While omin and omax are arguments, this function currently only converts
    to uint8
    '''
    cdef int x, y, b
    cdef int x_dim, y_dim, num_bands
    cdef float val
    cdef unsigned char new_val

    cdef unsigned short [:, :, :] src_view = src_ds
    dst_ds = np.empty_like(src_ds, dtype=c_uint8)
    cdef unsigned char [:, :, :] dst_view = dst_ds

    num_bands, x_dim, y_dim = np.shape(src_ds)

    for y in range(y_dim):
        for x in range(x_dim):
            for b in range(num_bands):
                val = src_view[b, x, y]
                if val == 0:
                    new_val = 0
                else:
                    if val < imin:
                        val = imin
                    elif val > imax:
                        val = imax
                    new_val = int(((val - imin) / (imax - imin)) * (omax - omin) + omin)
                dst_view[b, x, y] = new_val

    return np.copy(dst_view)


def _rescale_intensity_2d(src_ds, int imin, int imax, int omin, int omax):
    '''
    Rescales the input image intensity values
    While omin and omax are arguments, this function currently only converts
    to uint8
    '''
    cdef int x, y, b
    cdef int x_dim, y_dim, num_bands
    cdef float val
    cdef unsigned char new_val

    cdef unsigned short [:, :] src_view = src_ds
    dst_ds = np.empty_like(src_ds, dtype=c_uint8)
    cdef unsigned char [:, :] dst_view = dst_ds

    x_dim, y_dim = np.shape(src_ds)
    num_bands = 1

    for y in range(y_dim):
        for x in range(x_dim):
            val = src_view[x, y]
            if val == 0:
                new_val = 0
            else:
                if val < imin:
                    val = imin
                elif val > imax:
                    val = imax
                new_val = int(((val - imin) / (imax - imin)) * (omax - omin) + omin)
            dst_view[x, y] = new_val

    return np.copy(dst_view)


## NOT SURE IF THIS WORKS - NEED TO FINISH AND TEST, USE SKIMAGE IN MEAN TIME

def convert_to_hsv(src_ds, int imin, int imax, int omin, int omax):
    '''
    Rescales the input image intensity values.
    While omin and omax are arguments, this function currently only converts
    to uint8
    '''
    cdef int x, y, b
    cdef int x_dim, y_dim, num_bands
    cdef float val
    cdef unsigned char new_val

    cdef unsigned short [:, :, :] src_view = src_ds
    dst_ds = np.empty_like(src_ds, dtype=c_uint8)
    cdef unsigned char [:, :, :] dst_view = dst_ds

    num_bands, x_dim, y_dim = np.shape(src_ds)

    for y in range(y_dim):
        for x in range(x_dim):
            for b in range(num_bands):
                val = src_view[b, x, y]
                if val == 0:
                    new_val = 0
                else:
                    if val < imin:
                        val = imin
                    elif val > imax:
                        val = imax
                    new_val = int(((val - imin) / (imax - imin)) * (omax - omin) + omin)
                dst_view[b, x, y] = new_val

    return np.copy(dst_view)


def rgb2hsv(r, g, b):
    return _rgb2hsv(r,g,b)


def minmax(arr):
    return _minmax(arr)


cdef (float, float, float) _rgb2hsv(int r, int g, int b):
    cdef int min, max
    cdef float h, s, v, delta
    min, max = _minmax([r, g, b])

    v = max

    delta = max - min
    if max != 0:
        s = delta / max
    else:
        s = 0
        h = -1
        return h, s, v

    if r == max:
        h = (g - b) / delta
    elif g == max:
        h = 2 + (b - r) / delta
    else:
        h = 4 + (r - g) / delta

    h *= 60
    if h < 0:
        h += 360

    return h, s, v


# return type is a C struct of 2 values - this should be quick...
cdef (int, int) _minmax(arr):
    cdef int min = 255    # np.inf
    cdef int max = 0      # -np.inf
    cdef int i
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]
        if arr[i] > max:
            max = arr[i]
    return min, max
