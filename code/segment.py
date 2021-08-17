# title: Watershed Transform
# author: Nick Wright
# adapted from: Justin Chen, Arnold Song

import numpy as np
import gc
import warnings
from skimage import filters, morphology, feature, exposure, color
from skimage.future import graph
from skimage.util import img_as_ubyte
from scipy import ndimage
from ctypes import *
from lib import utils

# For Testing:
from skimage import segmentation
import matplotlib.image as mimg


def segment_image(input_data, image_type=False):
    '''
    Wrapper function that handles all of the processing to create watersheds
    '''
 
    #### Define segmentation parameters
    # High_threshold:
    # Low_threshold: Lower threshold for canny edge detection. Determines which "weak" edges to keep.
    #   Values above this amount that are connected to a strong edge will be marked as an edge.
    # Gauss_sigma: sigma value to use in the gaussian blur applied to the image prior to segmentation.
    #   Value chosen here should be based on the quality and resolution of the image
    # Feature_separation: minimum distance, in pixels, between the center point of multiple features. Use a lower value
    #   for lower resolution (.5m) images, and higher resolution for aerial images (~.1m).
    # These values are dependent on the type of imagery being processed, and are
    #   mostly empirically derived.
    # band_list contains the three bands to be used for segmentation
    if image_type == '1band':
        high_threshold = 0.15 * 255   ## Needs to be checked
        low_threshold = 0.05 * 255     ## Needs to be checked
        gauss_sigma = 1
        feature_separation = 1
        band_list = [0, 0, 0]
    elif image_type == '8band':
        high_threshold = 0.20 * 255    ## Needs to be checked
        low_threshold = 0.05 * 255      ## Needs to be checked
        gauss_sigma = 1.5
        feature_separation = 3
        band_list = [4, 2, 1]
    else:   #image_type == '3band'
        high_threshold = 0.15 # * 255
        low_threshold = 0.05 # * 255
        gauss_sigma = 1
        feature_separation = 2
        band_list = [2, 1, 0]

    segmented_data = watershed_transformation(input_data, band_list, low_threshold, high_threshold,
                                              gauss_sigma, feature_separation)

    # Method that provides the user an option to view the original image
    #  side by side with the segmented image.
    # print(np.amax(segmented_data))
    # image_data = np.array([input_data[band_list[2]],
    #                        input_data[band_list[1]],
    #                        input_data[band_list[0]]],
    #                       dtype=np.uint8)

    img = utils.create_composite(np.array([input_data[band_list[2]],
                                           input_data[band_list[1]],
                                           input_data[band_list[0]]],
                                          dtype=np.uint8))

    # ws_bound = segmentation.find_boundaries(segmented_data)
    # ws_display = utils.create_composite(image_data)
    # a = np.random.randint(0, 100)
    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/original1_{}.png'
    # mimg.imsave(save_name.format(a), ws_display, format='png')
    # #
    # ws_display[:, :, 0][ws_bound] = 240
    # ws_display[:, :, 1][ws_bound] = 80
    # ws_display[:, :, 2][ws_bound] = 80
    #
    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/seg1_{}.png'
    # mimg.imsave(save_name.format(a), ws_display, format='png')

    if np.amax(segmented_data) > 10:
        segmented_data = region_merge(img, segmented_data)

    # ws_bound = segmentation.find_boundaries(segmented_data)
    # ws_display = utils.create_composite(image_data)
    # # #
    # # # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/original_{}.png'
    # # # mimg.imsave(save_name.format(np.random.randint(0, 100)), ws_display, format='png')
    # # #
    # ws_display[:, :, 0][ws_bound] = 240
    # ws_display[:, :, 1][ws_bound] = 80
    # ws_display[:, :, 2][ws_bound] = 80
    #
    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/seg2_{}.png'
    # mimg.imsave(save_name.format(a), ws_display, format='png')

    return segmented_data


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


def region_merge(img, labels):

    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=20, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    return labels2


def watershed_transformation(image_data, band_list, low_threshold, high_threshold, gauss_sigma, feature_separation):
    '''
    Runs a watershed transform on the main dataset
        1. Create a gradient image using the sobel algorithm
        2. Adjust the gradient image based on given threshold and amplification.
        3. Find the local minimum gradient values and place a marker
        4. Construct watersheds on top of the gradient image starting at the
            markers.
    '''
    # If this block has no data, return a placeholder watershed.
    if np.amax(image_data[0]) <= 1:
        # We just need the dimensions from one band
        return np.zeros(np.shape(image_data[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value_image = color.rgb2hsv(np.array(utils.create_composite([image_data[band_list[2]],
                                                                     image_data[band_list[1]],
                                                                     image_data[band_list[0]]]),
                                             dtype=np.uint8))[:, :, 2]

    image_data = None

    # n = np.random.randint(0, 100)
    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/value_{}.png'
    # mimg.imsave(save_name.format(n), value_image, format='png', cmap='Greys_r')

    # Build a raster of detected edges to inform the creation of watershed seed points
    edge_image = edge_detect(value_image, band_list, gauss_sigma, low_threshold, high_threshold)
    # Build a raster of image gradient that will be the base for watershed expansion.
    grad_image = build_gradient(value_image, band_list, gauss_sigma)
    value_image = None

    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/edge_{}.png'
    # mimg.imsave(save_name.format(n), edge_image, format='png', cmap='Greys_r')

    # Find local minimum values in the edge image by inverting
    #   edge_image and finding the local maximum values
    inv_edge = np.empty_like(edge_image, dtype=np.uint8)
    np.subtract(255, edge_image, out=inv_edge)
    edge_image = None

    # Distance to the nearest detected edge
    distance_image = ndimage.distance_transform_edt(inv_edge)
    inv_edge = None

    # save_name = '/mnt/bamboo/OSSP_Tests/Debugging/dist_to_edge_{}.png'
    # mimg.imsave(save_name.format(n), distance_image, format='png', cmap='Greys_r')

    # Local maximum distance
    local_min = feature.peak_local_max(distance_image, min_distance=feature_separation,
                                       exclude_border=False, indices=False, num_peaks_per_label=1)
    distance_image = None

    markers = ndimage.label(local_min)[0]
    local_min = None

    # Build a watershed from the markers on top of the edge image
    im_watersheds = morphology.watershed(grad_image, markers)
    grad_image = None

    # Set all values outside of the image area (empty pixels, usually caused by
    #   orthorectification) to one value, at the end of the watershed list.
    # im_watersheds[empty_pixels] = np.amax(im_watersheds)+1
    gc.collect()
    return im_watersheds


def edge_detect(image_data, band_list, gauss_sigma, low_threshold, high_threshold):

    # Detect edges in the image with a canny edge detector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edge_image = img_as_ubyte(feature.canny(image_data, sigma=gauss_sigma,
                                                low_threshold=low_threshold, high_threshold=high_threshold))
    return edge_image


def build_gradient(image_data, band_list, gauss_sigma):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smooth_im_blue = ndimage.filters.gaussian_filter(image_data, sigma=gauss_sigma)
        scharr_im = filters.scharr(smooth_im_blue)
        grad_image = exposure.rescale_intensity(scharr_im, in_range='image', out_range=np.uint8)

    # Prevent the watersheds from 'leaking' along the sides of the image
    grad_image[:, 0] = grad_image[:, 1]
    grad_image[:, -1] = grad_image[:, -2]
    grad_image[0, :] = grad_image[1, :]
    grad_image[-1, :] = grad_image[-2, :]

    return grad_image