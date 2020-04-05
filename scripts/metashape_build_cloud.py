# This script processes a set of images and returns a set of camera poses 
# and a point cloud 3D model estimated by Agisoft Metashape. 
# 
# Desined for Agisoft Metashape Professional 1.6.4 
# (may be compatible with other versions: refer to Metashape Python API, https://www.agisoft.com/)

# How to run this script:
#    cd <folder_with_metashape_binary>
#    bash metashape.sh -r <path_to_this_script>


import os
import Metashape
import glob
from PIL import Image


def export_undistorted_photos(args, chunk):
    frames_path = os.path.join(args.in_dir, 'images_undistorted')
    os.makedirs(frames_path, exist_ok=True)

    for camera in chunk.cameras:
        for plane in camera.planes:
            img = plane.image()
            calib = plane.sensor.calibration

            img_undist = img.undistort(calib)
            img_undist.save(os.path.join(frames_path, camera.label + '.png'))


def get_images(args):
    image_dir = os.path.join(args.in_dir, 'images')
    if args.img_ext is not None:
        image_list = glob.glob('{}/*.{}'.format(image_dir, args.img_ext))
    else:
        image_list = (
            glob.glob('{}/*.jpg'.format(image_dir))
            + glob.glob('{}/*.png'.format(image_dir))
            + glob.glob('{}/*.jpeg'.format(image_dir))
            + glob.glob('{}/*.JPG'.format(image_dir))
            + glob.glob('{}/*.PNG'.format(image_dir))
            + glob.glob('{}/*.JPEG'.format(image_dir))
        )

    assert image_list, 'No images found in ' + image_dir

    image_list.sort()

    return image_list


def run(images, args):
    if not Metashape.app.activated:
        print('ERROR: Metashape is not activated. Please either purchase a license or use 30-day Professional trial.')
        exit()

    doc = Metashape.Document()

    doc.addChunk()
    chunk = doc.chunks[-1]

    print ("Adding photos...")
    chunk.addPhotos(images)

    print ("Matching photos...")
    #TODO choose Accuracy from args:
    chunk.matchPhotos(downscale=args.cam_downscale, generic_preselection=True, 
        reference_preselection=False)

    print("Aligning cameras...")
    chunk.alignCameras(adaptive_fitting=True)

    chunk.buildDepthMaps(downscale=args.pc_downscale)

    print ("Building a dense cloud...")
    chunk.buildDenseCloud()

    print("Saving the project...")
    doc.save(path=os.path.join(args.in_dir, "project.psz"))
    Metashape.app.update()

    print("Saving the point cloud...")
    chunk.exportPoints(path=os.path.join(args.in_dir, "point_cloud.ply"))

    print("Saving estimated cameras positions...")
    chunk.exportCameras(path=os.path.join(args.in_dir, "cameras.xml"),
        format=Metashape.CamerasFormat.CamerasFormatXML)

    print("Exporting undistorted photos...")
    export_undistorted_photos(args, chunk)

    print("Processing complete; please check directory {}.".format(args.in_dir))


def create_scene_config(images_list, args):
    img = Image.open(images_list[0])
    w, h = img.size

    with open(os.path.join(args.in_dir, 'scene.yaml'), 'w') as f:
        f.write('viewport_size: [{}, {}]\n'.format(w, h))
        f.write('intrinsic_matrix: cameras.xml\n')
        f.write('view_matrix: cameras.xml\n')
        f.write('pointcloud: point_cloud.ply')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('in_dir', type=str, 
                        help='Directory with input images.')
    
    parser.add_argument('--img_ext', type=str, 
                        help='Extension of images. If not specified, all images with .jpg, .jpeg, .png (case insensitive) will be taken.')
    parser.add_argument('--cam_downscale', type=int, default=1, 
                        help='Level of downscaling images before passing them to photo matching and cameras alignment procedures.'
                             '1 corresponds to no downscaling; 2, 3 or more define the downscaling factor by each side.')
    parser.add_argument('--pc_downscale', type=int, default=1, 
                        help='Level of downscaling images before passing them to depth map estimation and point cloud building procedures.'
                             '1 corresponds to no downscaling; 2, 3 or more define the downscaling factor by each side.')

    args = parser.parse_args()

    images_list = get_images(args)

    run(images_list, args)

    create_scene_config(images_list, args)