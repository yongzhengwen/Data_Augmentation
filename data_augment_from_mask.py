import os
import math
import cv2
import numpy as np
from multiprocessing.dummy import Pool

# Provide the directory path of the source images
imgdir = './src/'

# Provide the directory path of the masks (labels)
# Make sure that every image in the source folder has a corresponding mask the mask
# folder with the same name
maskdir = './label/'

# Provide the directory path of the output
savedir = "./out"

# The format of picture, jpg or png or anything else
image_format = 'png'

# How many augmented images from the original one image
agu_batch = 30

# The number of processes to initialize
WORKER_COUNT = 2

def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

    # hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    # hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    # hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);
    hsv[:, :, 0] = hsv[:, :, 0] * (0.95 + np.random.random() * 0.05)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.9 + np.random.random() * 0.1)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.9 + np.random.random() * 0.1)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR);
    return img


def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));


def r(val):
    return int(np.random.random() * val)


def AddNoiseSingleChannel(single):
    diff = 255 - single.max();
    noise = np.random.normal(0, 1 + r(3), single.shape);
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise;
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst


def addNoise(img):
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0]);
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1]);
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2]);
    return img;


def rotate_about_center(w, h, angle, scale=1.):
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return rot_mat, nw, nh


def work(imgpath, maskpath, savedir, batchsize):
    img = cv2.imread(imgpath)
    maskimg = cv2.imread(maskpath)
    if img is not None:
        print (imgpath)
    else:
        print ('error')
        return
    imgname = os.path.basename(imgpath).split('.')[0]
    cnt = 0
    for i in range(batchsize):

        rot_ang = r(5)
        if r(1) > 0.5:
            rot_ang = -rot_ang
        imgtp = img
        masktp = maskimg
        if rot_ang != 0:
            rot_mat_global, nw, nh = rotate_about_center(img.shape[1], img.shape[0], rot_ang)
            imgtp = cv2.warpAffine(img, rot_mat_global, (int(math.ceil(nw)), int(math.ceil(nh))),
                                   flags=cv2.INTER_LANCZOS4)
            masktp = cv2.warpAffine(maskimg, rot_mat_global, (int(math.ceil(nw)), int(math.ceil(nh))),
                                    flags=cv2.INTER_LANCZOS4)

        imgtp = tfactor(imgtp)
        imgtp = AddGauss(imgtp, r(2))
        imgtp = addNoise(imgtp)

        # Save the agumented images
        ss = "%s_%.6d.jpg" % (imgname, cnt)
        save_path = "%s/imgs/%s" % (savedir, ss)
        cv2.imwrite(save_path, imgtp)

        ss = "%s_%.6d.jpg" % (imgname, cnt)
        save_path = "%s/masks/%s" % (savedir, ss)
        cv2.imwrite(save_path, masktp)

        cnt += 1

    print (cnt)


def process(imgp, maskp, savedir):
    # thread_pool.apply_async(work, (imgp,xmlp,savedir))
    thread_pool.apply_async(work, (imgp, maskp, savedir, agu_batch))


if __name__ == "__main__":
    cmd = "mkdir %s" % savedir
    os.system(cmd)
    cmd = "mkdir %s/imgs" % savedir
    os.system(cmd)
    cmd = "mkdir %s/masks" % savedir
    os.system(cmd)
    thread_pool = Pool(WORKER_COUNT)
    print('Thread pool initialised with {} worker{}'.format(WORKER_COUNT, '' if WORKER_COUNT == 1 else 's'))

    data_list = os.listdir(imgdir)
    for indx in range(len(data_list)):
        d = data_list[indx]
        if d.find('.' + image_format) > 0:
            # img and mask need to have the same name
            imgpath = os.path.join(imgdir, d)
            maskpath = os.path.join(maskdir, d.split('.')[0] + '.' + image_format)
            print (imgpath, maskpath)
            if os.path.exists(maskpath):
                while True:
                    if (len(thread_pool._cache)) < WORKER_COUNT:
                        process(imgpath, maskpath, savedir)
                        break
                    else:
                        continue

    print("Waiting for workers to complete...")
    thread_pool.close()
    thread_pool.join()
