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
import matplotlib.pyplot as plt
import sys
sys.path.append("/data/d1/duanwei/FRUA/InsightFace/")
# from Learner import *
from config import get_config
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm


class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 10000
        self.alpha = 0.5 / 255
        self.de = (1 - 1 / self.num_iter)
        self.vis_long = 3

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

        conf = get_config()
        model1 = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        model1.load_state_dict(torch.load("/data/d1/duanwei/FRUA/InsightFace/work_space/models/model_ir_se50.pth",
                                         map_location=self.device))
        model1.eval().to(self.device)
        self.detector = detector
        self.model = model1



    def generate(self, time):

        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        rand = 0

        att_img = np.load('feature1.npy')
        vic_imgs = np.load('feature2.npy')

        # get victim feature
        org_img = []
        for i, image in enumerate(att_img):
            vic_img = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_img.div_(255).sub_(0.5).div_(0.5)
            org_img.append(vic_img)

        vic_feats = []
        for i, image in enumerate(vic_imgs):
            vic_image = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_image.div_(255).sub_(0.5).div_(0.5)
            vic_feat = self.model.forward(vic_image)
            vic_feat = vic_feat.cpu().detach().numpy().flatten()
            vic_feats.append(vic_feat)
        diff_w = np.load("/data/d1/duanwei/FRUA/some-resources-master/output/12-01-21-19diff.npy")
        diff_w = np.sign(diff_w) * np.minimum(abs(diff_w), 10)
        diff = torch.from_numpy(diff_w / 127.5).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        att_feats = []
        spc_feat = self.model.forward(org_img[rand])
        spc_feat = spc_feat.cpu().detach().numpy().flatten()
        for i,image in enumerate(org_img):
            att_feat = self.model.forward(image + diff)
            att_feat = att_feat.cpu().detach().numpy().flatten()
            att_feats.append(att_feat)

        num_succ = 0
        num_sup = 0
        num_fail = 0
        for i in range(100):
            if cos_sim(att_feats[i],spc_feat) > 0.3:
                num_succ = num_succ + 1
            elif cos_sim(att_feats[i],spc_feat) > cos_sim(att_feats[i],vic_feats[i]):
                num_sup = num_sup + 1
            else:
                num_fail += 1

        summary = "12010018  lp=10"
        with open('log/'+"log_attackratio.txt",'a') as fp2:
            print('summary %s' % (summary), file=fp2)
            print('num_success,num_super,num_fail: %d %d %d' %(num_succ,num_sup,num_fail), file=fp2)
            print('success_ratio %d\n' %(num_succ + num_sup), file=fp2)


        return


def main(args):

    tool = PyFAT()
    if args.device == 'cuda':
        tool.set_cuda()
    tool.load('assets')

    Time = datetime.datetime.now().strftime("%m-%d-%H-%M")

    tool.generate(Time)


if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        args = parser.parse_args()
        main(args)