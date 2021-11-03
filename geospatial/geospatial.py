import time
import subprocess


def reproject(src_filename, dst_filename, t_srs='EPSG:3413', verbose=False):
    """
    Reprojects the src file to the given epsg and saves the result
    in dst_filename.
    """
    if verbose:
        print("Reprojecting {} to {}".format(src_filename, dst_filename))

    warp_cmd = "gdalwarp -overwrite -multi -srcnodata 0 -dstnodata 0 " \
               "-r bilinear " \
               "-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER " \
               "-co PREDICTOR=2 -co ZLEVEL=9 " \
               "-wo NUM_THREADS=8 " \
               "--config GDAL_CACHEMAX 500 -wm 500 " \
               '-t_srs "{t_srs}" {fin} {fout}'

    cmd = warp_cmd.format(t_srs=t_srs, fin=src_filename, fout=dst_filename)

    time_start = time.time()
    #print(cmd)
    if verbose: print("Warping Image...")
    proc = subprocess.Popen(cmd, shell=False)#, stdin=subprocess.PIPE,
                            #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()
    proc = None
    addo_cmd = "gdaladdo {fin} 2 4 8 16 32 64"
    if verbose: print("Adding Overviews...")
    cmd = addo_cmd.format(fin=dst_filename)
    proc = subprocess.Popen(cmd, shell=False)#, stdin=subprocess.PIPE,
                            #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    time_end = time.time()
    duration = time_end - time_start

    print("Done. Time: {}".format(duration))

    return dst_filename