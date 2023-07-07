import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
import torch.nn.functional as F
from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils import norm_crop, logSaver


class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 5000
        self.alpha = 0.5/255

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
        img_shape = (112,112)
        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        
        # load face mask    
        # mask_np = cv2.resize(cv2.imread(osp.join(assets_path, 'mask.png')), img_shape) / 255
        # mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        # mask = F.interpolate(mask, img_shape).to(self.device)
        self.detector = detector
        self.model = model
        # self.mask = mask

    def size(self):
        return 1

    def generate(self, batch_image, n):
        h, w, c = batch_image[0].shape
        # assert len(im_a.shape) == 3
        # assert len(im_v.shape) == 3
        att_img =[]
        Matrix = []
        for i,image in enumerate(batch_image):
            bboxes, kpss = self.detector.detect(image, max_num=1)
            if bboxes.shape[0]==0:
                return image
            att_image, M = norm_crop(image, kpss[0], image_size=112)
            Matrix.append(M)
            att_image = att_image[:, :, ::-1]
            att_img.append(att_image)


        # get victim feature
        org_img = []
        vic_feats = []
        for i, image in enumerate(att_img,1):
            vic_img = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_img.div_(255).sub_(0.5).div_(0.5)
            vic_feat = self.model.forward(vic_img)
            org_img.append(vic_img)
            vic_feats.append(vic_feat)

        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        # process input
        diff =5/255 *  (-1 + 2 * np.random.random((112,112,3)))
        diff = torch.Tensor(diff.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        diff_ = diff.clone()
        diff.requires_grad = True
        for i in tqdm(range(self.num_iter)):
            self.model.zero_grad()
            # adv_images = att_img.clone()
          
            # get adv feature
            adv_feats = []
            for k,image in enumerate(org_img):
                adv_feat = self.model.forward(image + diff)
                adv_feats.append(adv_feat)

            # caculate loss and backward
            loss = 0
            for j in range(1,len(vic_feats)):
                loss = loss - F.cosine_embedding_loss(adv_feats[j], vic_feats[j], torch.tensor([1]).to(self.device)) \
                       + F.cosine_embedding_loss(adv_feats[0], vic_feats[j], torch.tensor([1]).to(self.device))
                # loss = loss - torch.mean(torch.square(adv_feats[j] - vic_feats[j]))
            # loss /= len(batch_image)
            loss.backward(retain_graph=True)

            print(loss.data)
            # for a in range(len(vic_feats)):
                # att_feat = adv_feats[a].cpu().detach().numpy().flatten()
                # org_feat = vic_feats[a].cpu().detach().numpy().flatten()
                # print(cos_sim(att_feat,org_feat))

            grad = diff.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = grad
            diff.data = diff.data - torch.sign(sum_grad) * self.alpha
            # att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - self.mask)
            diff.data = torch.clamp(diff.data, -10/255, 10/255)
            diff = diff.data.requires_grad_(True)
        # get diff and adv img
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5

        def inference(net, aimg):
            aimg = np.transpose(aimg, (2, 0, 1))
            aimg = torch.from_numpy(aimg).unsqueeze(0).float()
            aimg.div_(255).sub_(0.5).div_(0.5)
            aimg = aimg.cuda()
            feat = net(aimg).cpu().detach().numpy().flatten()
            return feat

        bboxes, kpss = self.detector.detect(batch_image[0], max_num=1)
        att_image_1, M = norm_crop(batch_image[0], kpss[0], image_size=112)
        vfeat1 = inference(self.model, att_image_1)
        for i in range(1,len(batch_image)):

            bboxes, kpss = self.detector.detect(batch_image[i], max_num=1)
            att_image, M = norm_crop(batch_image[i], kpss[0], image_size=112)

            feat = inference(self.model, att_image+diff)
            vfeat = inference(self.model, att_image)
            print(cos_sim(feat, vfeat),cos_sim(feat,vfeat1 ))
        # adv_image = []
        # for k in range(len(batch_image)):
        #     diff1 = diff
        #     diff1 = cv2.warpAffine(src=diff1, M=Matrix[k], dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        #     diff_bgr = diff1[:,:,::-1]
        #     image = batch_image[k] + diff_bgr
        #     adv_image.append(image)
        return diff

def main(args):

    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = PyFAT()
    if args.device=='cuda':
        tool.set_cuda()
    tool.load('assets')

    batch_image = []
    for idname in range(1, 9):
        str_idname = "%03d"%idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic = osp.join(iddir, '1.png')
        origin_att_img = cv2.imread(att)
        origin_vic_img = cv2.imread(vic)
        batch_image.append(origin_att_img)
        batch_image.append(origin_vic_img)


    # ta = datetime.datetime.now()
    diff = tool.generate(batch_image, 0)
    # tb = datetime.datetime.now()
    #print( (tb-ta).total_seconds() )
    # save_name = '{}.png'.format(i)
    cv2.imwrite(save_dir + '/' + 'diff.png', diff)
        

if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        args = parser.parse_args()
        main(args)

