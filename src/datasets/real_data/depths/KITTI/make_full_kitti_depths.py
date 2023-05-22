import os
import subprocess
import traceback
import cv2
import skimage
import scipy
import numpy as np
import tqdm
from scipy.sparse.linalg import spsolve
import signal
from PIL import Image

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in tqdm.tqdm(range(W)):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output

is_slurm = False
command_prefix = "srun -c 1" if is_slurm else ''

def store_full_depth(rgb_path, depth_path, save_path):
    image = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, -1)
    output = fill_depth_colorization(image, depth, alpha=1).astype(np.uint16)
    cv2.imwrite(save_path, output)

"""
creating a full depth map from the given sparse depth map file  
"""
if __name__ == '__main__':

    # global images
    images = []
    base_path = '.'  # 'datasets/real_data/depths/KITTI'
    start_path = f'{base_path}/rgbs'
    workers = []
    def recursive(path):
        global images
        for dir in os.listdir(path):
            if 'image_0' in dir and dir != 'image_03':
                continue
            curr_path = f"{path}/{dir}"
            if os.path.isdir(curr_path):
                recursive(curr_path)
            elif os.path.isfile(curr_path):
                try:
                    rgb_path = curr_path
                    folder_name = curr_path.split("/")[-4]
                    file_name = curr_path.split("/")[-1]
                    if file_name.split(".")[-1] != 'png':
                        continue
                    depth_path = f"{base_path}/depths/{folder_name}/proj_depth/groundtruth/image_03/{file_name}"
                    if not os.path.isfile(depth_path):
                        print("does not exist")
                        continue
                    dir_path = f"{base_path}/full_depths/{folder_name}/proj_depth/groundtruth/image_03"
                    os.makedirs(dir_path, exist_ok=True)
                    save_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(save_path):
                        # print("already exists")
                        continue
                    store_full_depth(rgb_path, depth_path, save_path)
                    # images += [(rgb_path, depth_path)]
                except Exception as e:
                    print(f"path {curr_path}", e)
                    print(traceback.format_exc())
                    print()

    recursive(start_path)

