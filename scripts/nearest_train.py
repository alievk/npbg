import numpy as np
import xml.etree.ElementTree as ET
import cv2
import argparse


def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def get_valid_matrices(mlist):
    ilist = []
    vmlist = []
    for i, m in enumerate(mlist):
        if np.isfinite(m).all():
            ilist.append(i)
            vmlist.append(m)

    return vmlist, ilist


def extrinsics_from_view_matrix(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm, ids = get_valid_matrices(vm)

    return vm, ids


def get_vec(view_mat):
    view_mat = view_mat.copy()
    rvec0 = cv2.Rodrigues(view_mat[:3, :3])[0].flatten()
    t0 = view_mat[:3, 3]
    return rvec0, t0

def nearest_train(view_mat, test_pose, p=0.05):
    dists = []
    angs = []
    test_rvec, test_t = get_vec(test_pose)
    for i in range(len(view_mat)):
        rvec, t = get_vec(view_mat[i])
        dists.append(
            np.linalg.norm(test_t - t)
        )
        angs.append(
            np.linalg.norm(test_rvec - rvec)
        )
    angs_sort = np.argsort(angs)
    angs_sort = angs_sort[:int(len(angs_sort) * p)]
    dists_pick = [dists[i] for i in angs_sort]
    ang_dist_i = angs_sort[np.argmin(dists_pick)]
    return ang_dist_i #, angs_sort[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view_mat', help='xml or txt')
    parser.add_argument('--pose', help='test pose, 4x4 or Nx4x4 txt file')
    parser.add_argument('--origin', help='(optional) model origin pose, 4x4 or Nx4x4 txt file')
    parser.add_argument('--gt')
    args = parser.parse_args()

    if args.view_mat[-3:] == 'xml':
        view_mat, labels = extrinsics_from_xml(args.view_mat)
    else:
        view_mat, labels = extrinsics_from_view_matrix(args.view_mat)

    test_pose = np.loadtxt(args.pose).reshape(-1, 4, 4)

    if args.origin:
        origin = np.loadtxt(args.origin)
    else:
        origin = np.eye(4)

    for tp in test_pose: 
        tp = np.linalg.inv(origin) @ tp
        nearest_i = nearest_train(view_mat, tp)
        label = labels[nearest_i]

        if args.gt:
            path = args.gt.replace('*', str(label))
            print(path)
        else:
            print(label)
