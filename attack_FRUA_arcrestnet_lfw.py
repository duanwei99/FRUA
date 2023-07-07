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
sys.path.append("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master/")# from Learner import *
from pathlib import Path
from PIL import Image
from config import get_config
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms


class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 5000
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
        # model = iresnet.iresnet50()
        # weight = osp.join(assets_path, 'w600k_r50.pth')
        # model.load_state_dict(torch.load(weight, map_location=self.device))
        # conf = get_config(use_mobilfacenet = False)
        # model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        # model.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/InsightFace/work_space/models/model_ir_se50.pth", map_location=self.device))
        # model.eval().to(self.device)
        # self.detector = detector
        # self.model = model

    def size(self):
        return 1

    def tanh_space(self, x):
        return torch.tanh(x)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))


    def generate(self, time):

        print(time)
        """"start_seed = 0"""

        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        # def getFileList(dir, Filelist, ext=None):
        #     """
        #     获取文件夹及其子文件夹中文件列表
        #     输入 dir：文件夹根目录
        #     输入 ext: 扩展名
        #     返回： 文件路径列表
        #     """
        #     newDir = dir
        #     if os.path.isfile(dir):
        #         if ext is None:
        #             Filelist.append(dir)
        #         else:
        #             if ext in dir[-3:]:  # jpg为-3/py为-2
        #                 Filelist.append(dir)
        #
        #     elif os.path.isdir(dir):
        #         for s in os.listdir(dir):
        #             newDir = os.path.join(dir, s)
        #             getFileList(newDir, Filelist, ext)
        #     return Filelist

        rand = 3


        # img = []
        # imglist = getFileList("/data/hdd/duanwei/dataset/cfp-dataset/Data/Images", [], 'jpg')
        # resize2 = transforms.Resize([112, 112])
        #
        # for imgpath in tqdm(imglist):
        #     # imgname = os.path.splitext(os.path.basename(imgpath))[0]
        #     img1 = Image.open(imgpath)
        #     img1 = resize2(img1)
        #     img1 = np.array(img1)
        #     if img1.shape == (112,112,3):
        #         img.append(img1)
        # get victim feature
        img = np.load('dataset/feature_age_stdall.npy')
        org_img = []
        vic_feats = []
        print("data to tensor")
        for i, image in tqdm(enumerate(img)):
            vic_image = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_image.div_(255).sub_(0.5).div_(0.5)
            org_img.append(vic_image)
            vic_feat = self.model.forward(vic_image)
            vic_feat = vic_feat.cpu().detach().numpy().flatten()
            vic_feats.append(vic_feat)

        wolf_img = org_img[rand].clone()
        # wolf_img= torch.Tensor(org_img[0].copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # wolf_img.div_(255).sub_(0.5).div_(0.5)

        diff_zeros = torch.zeros_like(wolf_img)
        w = self.inverse_tanh_space(wolf_img).detach()
        w.requires_grad = True

        MSELoss = nn.MSELoss(reduction='sum')
        Flatten = torch.nn.Flatten()

        w1 =10
        w2 = 20000
        g, decay_factor = 0, 0.5
        # optimizer = optim.Adam([w],lr=0.001)
        # a = tqdm(range(self.num_iter), file=sys.stdout)
        # loss1 = []
        # loss2 = []
        # loss_total = []
        # delta = 10 / 127.5
        # iter_num = 0
        # L2 = 'L2 = MSELoss(Flatten(diff_w), Flatten(diff_zeros))'
        # for i in enumerate(a):
        #     iter_num +=1
        #     self.model.zero_grad()
        #
        #     # get adv feature
        #     diff_w = self.tanh_space(w)
        #     wolf_feats = []
        #     for i,image in enumerate(org_img[rand+1:rand+13]):
        #         wolf_image = torch.clamp(image + diff_w,-1,1)
        #         wolf_feat= self.model.forward(wolf_image)
        #         wolf_feats.append(wolf_feat)
        #     org_feat = self.model.forward(org_img[rand])
        #
        #     # caculate loss and backward
        #
        #     loss = 0
        #     for j in range(0, len(wolf_feats)):
        #         loss = loss + F.cosine_embedding_loss(wolf_feats[j], org_feat, torch.tensor([1]).to(self.device))
        #     l2_loss = MSELoss(Flatten(diff_w), Flatten(diff_zeros))
        #     # l2_loss = torch.max(diff_w)
        #     cost = w1 * l2_loss + w2 * loss
        #     loss1.append(w1 * l2_loss.item())
        #     loss2.append(w2 * loss.item())
        #     loss_total.append(cost.item())
        #     print(w1 * l2_loss.data, w2 * loss.data)
        #
        #     optimizer.zero_grad()
        #     cost.backward(retain_graph=True)
        #     optimizer.step()
        #     # loss.backward(retain_graph=True)
        #
        #     a.desc= "loss:{:.3f}".format(cost)
        #
        #     # grad = diff.grad.data.clone()
        #     # grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
        #     # sum_grad = grad
        #     # diff.data = diff.data - torch.sign(sum_grad) * self.alpha
        #     # grad = diff.grad.data.clone()
        #     # g = decay_factor * g + grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
        #     # diff.data = diff.data - torch.sign(g) * self.alpha
        #     # att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - self.mask)
        #     # diff.data = torch.clamp(diff.data, -10 / 255, 10 / 255)
        #     # diff = diff.data.requires_grad_(True)
        #     if iter_num == 2000:
        #         diff_w = torch.clip(diff_w, -delta, delta)
        # # get diff and adv img
        # # print(loss.data,l2_loss.data)
        # # diff = diff_w
        #
        # diff = torch.clip(diff_w,-delta,delta)

        diff = np.load("/data/hdd/duanwei/FRUA/some-resources-master/output/04-01-13-43diff.npy")
        diff = diff / 127.5
        diff = torch.Tensor(diff.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_feats = []
        # spc_feat = self.model.forward(org_img[rand])
        # spc_feat = spc_feat.cpu().detach().numpy().flatten()
        spc_feat = np.load("dataset/spec_mob_age.npy")
        print("verify begin")
        for i,image in tqdm(enumerate(org_img)):
            att_feat = self.model.forward(image + diff)
            att_feat = att_feat.cpu().detach().numpy().flatten()
            att_feats.append(att_feat)

        num_succ = 0
        num_sup = 0
        num_fail = 0
        for i in range(len(org_img)):
            if cos_sim(att_feats[i],spc_feat) > 0.3:
                num_succ = num_succ + 1
            elif cos_sim(att_feats[i],spc_feat) > cos_sim(att_feats[i],vic_feats[i]):
                num_sup = num_sup + 1
            else:
                num_fail += 1

        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        np.save('output/'+str(time)+'diff.npy',diff)
        summary = "test InsightFace Model(resnet -IR) detla=10/127.5 iter2000 clip LFW modify w1:w2"
        # with open('log/'+str(time)+"_log.txt",'w') as fp2:
        print('summary %s' % (summary))
        print('start_seed %d' %(rand))
        print('num_success,num_super,num_fail: %d %d %d' %(num_succ,num_sup,num_fail))
        print('success_ratio %f' %((num_succ + num_sup)/(num_succ + num_sup+num_fail)))
        # print(L2,file=fp2)
        print('w1 and w2 %d %d   the ratio is   %f:' %(w1,w2,w2/w1))
        print('the diff max and min is %f %f' %(diff.max(),diff.min()))
            # print(self.num_iter, file=fp2)
            # for i, v in enumerate(loss_total):
            #     print(loss1[i], loss2[i], v, file=fp2)

        # visible = np.ones((112,112,3)) + 127
        image = [[0] for i in range(self.vis_long)]
        for i in range(self.vis_long):
            image[i] = diff + img[i+1]
            image[i] = image[i][:,:,::-1]
            image[i][image[i] < 0] = 0
            image[i][image[i] > 255] = 255

        return diff[:,:,::-1] ,image


def main(args):
    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = PyFAT()
    if args.device == 'cuda':
        tool.set_cuda()
    tool.load('assets')

    Time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    # ta = datetime.datetime.now()
    diff,vis_diff = tool.generate(Time)

    # tb = datetime.datetime.now()
    # print( (tb-ta).total_seconds() )
    # save_name = '{}.png'.format(i)
    # cv2.imwrite(save_dir + '/' + str(Time)+args.core+'diff.png', diff)
    # for i in range(3):
    #     cv2.imwrite(save_dir + '/' + str(Time)+args.core+'add_diff'+str(i+1)+'.png', vis_diff[i])


if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        parser.add_argument('--core',help='',type=str,default='mask')
        args = parser.parse_args()
        main(args)