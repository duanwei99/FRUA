import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
import numpy as np
import datetime
import random
import torch
import glob
import time
import cv2
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils import norm_crop, logSaver



class data_loder:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = True



    def set_cuda(self):
        self.is_cuda = True
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        img_shape = (112, 112)
        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.detector = detector
        self.model = model


    def generate(self, batch_image,vic_images, n):
        h, w, c = batch_image[0].shape
        # assert len(im_a.shape) == 3
        # assert len(im_v.shape) == 3
        att_img = []
        for i, image in enumerate(batch_image):
            # bboxes, kpss = self.detector.detect(image, max_num=1)
            # if bboxes.shape[0] == 0:
            #     return image
            # att_image, M = norm_crop(image, kpss[0], image_size=112)
            # Matrix.append(M)
            att_image = image[:, :, ::-1]
            att_img.append(att_image)
        np.save('feature1_mtcnn.npy',att_img)

        vic_img = []
        for i, image in enumerate(vic_images):
            # bboxes, kpss = self.detector.detect(image, max_num=1)
            # if bboxes.shape[0] == 0:
            #     return image
            # att_image, M = norm_crop(image, kpss[0], image_size=112)
            # Matrix.append(M)
            att_image = image[:, :, ::-1]
            vic_img.append(att_image)
        np.save('feature2_mtcnn.npy', vic_img)

def main(args):
    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = data_loder()
    if args.device == 'cuda':
        tool.set_cuda()
    tool.load('assets')

    batch_image = []
    vic_images = []
    for idname in range(100):
        # str_idname = "%03d" % idname
        iddir = osp.join('/data/d1/duanwei/FRUA/data/img_subset', str(idname))
        filelist = os.listdir(iddir)
        att_image=os.path.join(iddir,filelist[0])
        att_img = cv2.imread(att_image)
        batch_image.append(att_img)
        vic_image = os.path.join(iddir, filelist[1])
        vic_img = cv2.imread(vic_image)
        vic_images.append(vic_img)


    tool.generate(batch_image,vic_images, 0)




if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        parser.add_argument('--core',help='',type=str,default='mask')
        args = parser.parse_args()
        main(args)