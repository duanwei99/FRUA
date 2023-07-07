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
from torchvision.transforms import Resize
sys.path.append("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master/")
# from Learner import *
from config import get_config
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
sys.path.append("/data/hdd/duanwei/FRUA/sphereface_pytorch-master")
import net_sphere



class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 6000
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
        self.detector = detector
        img_shape = (112, 112)
        # model = iresnet.iresnet50()
        # weight = osp.join(assets_path, 'w600k_r50.pth')
        # model.load_state_dict(torch.load(weight, map_location=self.device))
        conf = get_config(use_mobilfacenet = False)
        model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        model.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/InsightFace/work_space/models/model_ir_se50.pth",
                                         map_location=self.device))
        model.eval().to(self.device)
        self.model1 = model

        conf = get_config(use_mobilfacenet = True)
        model = MobileFaceNet(conf.embedding_size).to(conf.device)
        model.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master/work_space/models/model_mobilefacenet.pth", map_location=self.device))
        model.eval().to(self.device)
        self.model2 = model

        net = getattr(net_sphere, 'sphere20a')()
        net.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/sphereface_pytorch-master/model/sphere20a_20171020.pth"))
        net.cuda()
        net.eval()
        net.feature = True
        self.model3 = net




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

        def cosine_decay_with_warmup(global_step,
                                     warmup_end=5000,
                                     total_steps=6000,
                                     warmup_start=1700,
                                     warmup_steps=2000,
                                     hold_base_rate_steps=2000):
            """
            参数：
                    global_step: 上面定义的Tcur，记录当前执行的步数。
                    learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
                    total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
                    warmup_learning_rate: 这是warm up阶段线性增长的初始值
                    warmup_steps: warm_up总的需要持续的步数
                    hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
            """
            if total_steps < warmup_steps:
                raise ValueError('total_steps must be larger or equal to '
                                 'warmup_steps.')
            # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
            learning_rate = warmup_start + 0.5 * (warmup_end - warmup_start) * (1 + np.cos(
                np.pi * (global_step - warmup_steps - hold_base_rate_steps) / float(
                    total_steps - warmup_steps - hold_base_rate_steps)))
            # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
            if hold_base_rate_steps > 0:
                learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                         learning_rate, warmup_end)
            if warmup_steps > 0:
                if warmup_end < warmup_start:
                    raise ValueError('learning_rate_base must be larger or equal to '
                                     'warmup_learning_rate.')
                # 线性增长的实现
                slope = (warmup_end - warmup_start) / warmup_steps
                warmup_rate = slope * global_step + warmup_start
                # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
                learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                         learning_rate)
            return np.where(global_step > total_steps, 0.0, learning_rate)

        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        rand = 0

        att_imgs = np.load('feature1.npy')
        vic_imgs = np.load('feature2.npy')
        # att_imgs_sphere = np.load('feature1_sphere.npy')
        # vic_imgs_sphere = np.load('feature2_sphere.npy')
        # image = att_imgs[1]
        # image_reszie = att_imgs_sphere[1]
        # fig_new = plt.figure()
        # ax = fig_new.add_subplot(121)
        # ax.imshow(image)
        # ax = fig_new.add_subplot(122)
        # ax.imshow(image_reszie)
        # plt.show()
        # get victim feature


        # get victim feature
        torch_resize = Resize([112,96])
        org_img = []
        org_img_sphere = []
        for i, image in enumerate(att_imgs):
            vic_img = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_img.div_(255).sub_(0.5).div_(0.5)
            im1_resize = torch_resize(vic_img)
            org_img.append(vic_img)
            org_img_sphere.append(im1_resize)

        # vic_feats = []
        # for i, image in enumerate(vic_imgs):
        #     vic_image = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        #     vic_image.div_(255).sub_(0.5).div_(0.5)
        #     vic_feat = self.model.forward(vic_image)
        #     vic_feat = vic_feat.cpu().detach().numpy().flatten()
        #     vic_feats.append(vic_feat)

        wolf_img = ((20 * torch.rand(org_img[rand].shape) - 10) / 255).to(self.device)

        diff_zeros = torch.zeros_like(wolf_img)
        w = self.inverse_tanh_space(wolf_img).detach()
        w.requires_grad = True

        MSELoss = nn.MSELoss(reduction='sum')
        Flatten = torch.nn.Flatten()

        w1 =100
        w2 = 2100
        g, decay_factor = 0, 0.5
        optimizer = optim.Adam([w],lr=0.001)
        a = tqdm(range(self.num_iter), file=sys.stdout)
        loss1 = []
        loss2 = []
        loss_total = []
        delta = 10 / 127.5
        iter_num = 0


        L2 = 'L2 = MSELoss(Flatten(diff_w), Flatten(diff_zeros))'
        for i in enumerate(a):
            iter_num +=1
            self.model1.zero_grad()
            self.model2.zero_grad()
            self.model3.zero_grad()

            # get adv feature
            diff_w = self.tanh_space(w)
            wolf_feats1 = []
            for k,image in enumerate(org_img[rand+1:rand+25]):
                wolf_image = torch.clamp(image + diff_w,-1,1)
                wolf_feat= self.model1.forward(wolf_image)
                wolf_feats1.append(wolf_feat)
            org_feat1 = self.model1.forward(org_img[rand])
            wolf_feats2 = []
            for k, image in enumerate(org_img[rand + 1:rand + 25]):
                wolf_image = torch.clamp(image + diff_w, -1, 1)
                wolf_feat = self.model2.forward(wolf_image)
                wolf_feats2.append(wolf_feat)
            org_feat2 = self.model2.forward(org_img[rand])
            wolf_feats3 = []
            for k, image in enumerate(org_img_sphere[rand + 1:rand + 25]):
                wolf_image = torch.clamp(image + torch_resize(diff_w), -1, 1)
                wolf_feat = self.model3.forward(wolf_image)
                wolf_feats3.append(wolf_feat)
            org_feat3 = self.model3.forward(org_img_sphere[rand])

            # caculate loss and backward

            loss = 0
            for j in range(0, len(wolf_feats1)):
                loss = loss + F.cosine_embedding_loss(wolf_feats1[j], org_feat1, torch.tensor([1]).to(self.device))
            for j in range(0, len(wolf_feats2)):
                loss = loss + F.cosine_embedding_loss(wolf_feats2[j], org_feat2, torch.tensor([1]).to(self.device))
            # for j in range(0, len(wolf_feats3)):
            #     loss = loss + F.cosine_embedding_loss(wolf_feats3[j], org_feat3, torch.tensor([1]).to(self.device))
            l2_loss = MSELoss(Flatten(diff_w), Flatten(diff_zeros))
            # l2_loss = torch.max(diff_w)
            w2 = torch.tensor(3000 + (1000*np.cos(np.pi*iter_num/1000))).to(self.device)
            # w2 = torch.tensor(cosine_decay_with_warmup(global_step = iter_num)).to(self.device)
            cost = w1 * l2_loss + w2 * loss
            loss1.append(w1 * l2_loss.item())
            loss2.append(w2 * loss.item())
            loss_total.append(cost.item())
            print(w1 * l2_loss.data, w2 * loss.data,w2)

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()
            # loss.backward(retain_graph=True)

            a.desc= "loss:{:.3f}".format(cost)

            if iter_num == 2000:
                diff_w = torch.clip(diff_w, -delta, delta)

        diff = torch.clip(diff_w,-delta,delta)
        # diff = np.load("/data/hdd/duanwei/FRUA/some-resources-master/output/03-12-19-44diff.npy")
        # diff = diff / 127.5
        # diff = torch.Tensor(diff.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # att_feats = []
        # spc_feat = self.model.forward(org_img[rand])
        # spc_feat = spc_feat.cpu().detach().numpy().flatten()
        # # spc_feat = np.load("spec_feat.npy")
        # for _,image in enumerate(org_img):
        #     att_feat = self.model.forward(image + diff)
        #     att_feat = att_feat.cpu().detach().numpy().flatten()
        #     att_feats.append(att_feat)
        #
        # num_succ = 0
        # num_sup = 0
        # num_fail = 0
        # for i in range(100):
        #     if cos_sim(att_feats[i],spc_feat) > 0.3:
        #         num_succ = num_succ + 1
        #     elif cos_sim(att_feats[i],spc_feat) > cos_sim(att_feats[i],vic_feats[i]):
        #         num_sup = num_sup + 1
        #     else:
        #         num_fail += 1

        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5

        summary = "cross Model without sphereface"
        np.save('stage2/output/' + str(time)+summary + 'diff.npy', diff)
        # with open('stage2/log/'+str(time)+summary +"_log.txt",'w') as fp2:
        #     print('summary %s' % (summary), file=fp2)
        #     print('start_seed %d' %(rand), file=fp2)
        #     print('num_success,num_super,num_fail: %d %d %d' %(num_succ,num_sup,num_fail), file=fp2)
        #     print('success_ratio %d' %(num_succ + num_sup), file=fp2)
        #     # print(L2,file=fp2)
        #     print('w1 and w2 %d %d   the ratio is   %f:' %(w1,w2,w2/w1), file=fp2)
        #     print('the diff max and min is %f %f' %(diff.max(),diff.min()), file=fp2)



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
    tool.generate(Time)



if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        parser.add_argument('--core',help='',type=str,default='mask')
        args = parser.parse_args()
        main(args)