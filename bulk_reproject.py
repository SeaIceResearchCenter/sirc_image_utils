import os
import argparse
from geospatial import geospatial


def bulk_reproject():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help='''directory path containing images to be reprojected''')
    parser.add_argument("-o", "--output_dir", type=str, default="default",
                        help="directory to place output results.")
    parser.add_argument("-t_srs", "--t_srs", type=str, default="3413",
                        help="target srs for reprojection")  

    args = parser.parse_args()

    # System filepath that contains the directories or files for batch processing
    user_input = args.input_dir
    if os.path.isdir(user_input):
        src_dir = user_input
        src_file = ''
    elif os.path.isfile(user_input):
        src_dir, src_file = os.path.split(user_input)
    else:
        raise IOError('Invalid input')

    dst_dir = args.output_dir
    if dst_dir == 'default':
        dst_dir = src_dir
    else:
        # Make the output directory if it doesnt already exist
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

    t_srs = args.t_srs

    # Loop through contents of the given directory
    for file in os.listdir(src_dir):

        # Skip hidden files
        if file[0] == '.':
            continue

        image_name,ext = os.path.splitext(file)
        # Check that the file is .tif or .jpg format
        ext = ext.lower()
        if ext != '.tif':
            continue
        
        input_file = os.path.join(src_dir, file)
        
        output_name = image_name + '_{}'.format(t_srs) + ext
        output_file = os.path.join(dst_dir, output_name)

        geospatial.reproject(input_file, output_file)
        

if __name__ == '__main__':
    bulk_reproject()