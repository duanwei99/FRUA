import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

        # load face mask
        # mask_np = cv2.resize(cv2.imread(osp.join(assets_path, 'mask.png')), img_shape) / 255
        # mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        # mask = F.interpolate(mask, img_shape).to(self.device)
        self.detector = detector
        self.model = model
        # self.mask = mask

    def size(self):
        return 1

    def tanh_space(self, x):
        return torch.tanh(x)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def generate(self, batch_image,vic_images, n):
        h, w, c = batch_image[0].shape
        # assert len(im_a.shape) == 3
        # assert len(im_v.shape) == 3
        att_img = []
        Matrix = []
        for i, image in enumerate(batch_image[:len(batch_image)-2]):
            bboxes, kpss = self.detector.detect(image, max_num=1)
            if bboxes.shape[0] == 0:
                return image
            att_image, M = norm_crop(image, kpss[0], image_size=112)
            # Matrix.append(M)
            att_image = att_image[:, :, ::-1]
            att_img.append(att_image)

        # get victim feature
        org_img = []
        # vic_feats = []
        for i, image in enumerate(att_img):
            vic_img = torch.Tensor(image.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            vic_img.div_(255).sub_(0.5).div_(0.5)
            # vic_feat = self.model.forward(vic_img)
            org_img.append(vic_img)
            # vic_feats.append(vic_feat)

        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        # process input
        # diff = 5 / 255 * (np.random.random((112, 112, 3)))
        # diff = torch.Tensor(diff.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # diff_ = diff.clone()


        # image[image < 0] = 0
        # image = image.astype(np.uint8)
        # cv2.imshow("Image",image)
        # cv2.waitKey(0) == 'q'

        # plt.figure("Image")
        # plt.imshow(image)
        # plt.axis("on")
        # plt.title('image')
        # plt.show()

        #randomly masking
        # mask_ratio = 0.5
        # N,D,L,W= wolf_img.shape  # batch, length, dim
        # len_keep = int(L * (1 - mask_ratio))
        #
        # noise = torch.rand(N, L, device=wolf_img.device)  # noise in [0, 1]
        #
        # # sort noise for each sample
        # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        #
        # # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # wolf_masked = torch.gather(wolf_img, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, D ,1,1))
        #
        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=wolf_img.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        # for i in range(112):
        #     # rand = random.randint(0,111)
        #     # att_img[0][rand] = 0
        #     if i % 3 != 0:
        #         att_img[0][i] = 0
        #     elif i % 4 != 0:
        #         att_img[0] = att_img[0].transpose(1,0,2)
        #         att_img[0][i] = 0        #         att_img[0] = att_img[0].transpose(1, 0, 2)
        wolf_img = torch.Tensor(att_img[0].copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        image_wolf = wolf_img.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
        image_wolf[image_wolf < 0] = 0
        image_wolf = image_wolf.astype(np.uint8)
        # cv2.imshow("Image_1", image_wolf)
        nowTime = datetime.datetime.now().strftime("%m-%d-%H-%M")
        # cv2.imwrite('output/'+str(nowTime)+'masking'+'.png',image_wolf)
        # cv2.waitKey(0) == 'q'

        wolf_img.div_(255).sub_(0.5).div_(0.5)
        diff_zeros = torch.zeros_like(wolf_img)
        w = self.inverse_tanh_space(wolf_img).detach()
        w.requires_grad = True

        MSELoss = nn.MSELoss(reduction='sum')
        Flatten = torch.nn.Flatten()

        w1 = 10
        w2 = 1000
        g, decay_factor = 0, 0.5
        optimizer = optim.Adam([w],lr=0.001)
        a = tqdm(range(self.num_iter), file=sys.stdout)
        loss1 = []
        loss2 = []
        loss_total = []
        for i in enumerate(a):
            self.model.zero_grad()
            # adv_images = att_img.clone()
            # w = w * self.de
            # get adv feature
            diff_w = self.tanh_space(w)

            wolf_feats = []
            for i,image in enumerate(org_img):
                wolf_feat= self.model.forward(image + diff_w)
                wolf_feats.append(wolf_feat)
            org_feat = self.model.forward(org_img[0])

            # caculate loss and backward
            loss = 0
            for j in range(0, len(wolf_feats)):
                loss = loss + F.cosine_embedding_loss(wolf_feats[j], org_feat, torch.tensor([1]).to(self.device))
            # l2_loss = MSELoss(Flatten(diff_w), Flatten(diff_zeros))
            l2_loss = torch.pow(Flatten(diff_w),4).sum()
            # l2_loss = torch.max(diff_w)
            cost = w1 *l2_loss + w2 *loss
            loss1.append(w1 * l2_loss.item())
            loss2.append(w2 * loss.item())
            loss_total.append(cost.item())
            print(w1*l2_loss.data, w2*loss.data)

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()
            # loss.backward(retain_graph=True)

            a.desc= "loss:{:.3f}".format(cost)

            # grad = diff.grad.data.clone()
            # grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # sum_grad = grad
            # diff.data = diff.data - torch.sign(sum_grad) * self.alpha
            # grad = diff.grad.data.clone()
            # g = decay_factor * g + grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            # diff.data = diff.data - torch.sign(g) * self.alpha
            # att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - self.mask)
            # diff.data = torch.clamp(diff.data, -10 / 255, 10 / 255)
            # diff = diff.data.requires_grad_(True)
        # get diff and adv img
        # print(loss.data,l2_loss.data)
        diff = diff_w
        # diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5

        def inference(net, aimg):
            # aimg = cv2.cvtColor(aimg,cv2.COLOR_BGR2RGB)
            aimg = np.transpose(aimg, (2, 0, 1))
            aimg = torch.from_numpy(aimg).unsqueeze(0).float()
            aimg.div_(255).sub_(0.5).div_(0.5)
            aimg = aimg.cuda()
            feat = net(aimg).cpu().detach().numpy().flatten()
            return feat

        # bboxes, kpss = self.detector.detect(batch_image[0], max_num=1)
        # att_image_1, M = norm_crop(batch_image[0], kpss[0], image_size=112)
        # vfeat1 = inference(self.model, att_image_1)
        vics_feat = []
        for i in range(0, len(batch_image)):
            bboxes, kpss = self.detector.detect(vic_images[i], max_num=1)
            att_image, M = norm_crop(vic_images[i], kpss[0], image_size=112)
            vic_feat = inference(self.model, att_image)
            vics_feat.append(vic_feat)
        for i in range(0, len(batch_image)):
            bboxes, kpss = self.detector.detect(batch_image[i], max_num=1)
            att_image, M = norm_crop(batch_image[i], kpss[0], image_size=112)

            att_feat = inference(self.model, att_image + diff)
            org_feat = inference(self.model, att_image)
            print("The {} th verification".format(i))
            for j in range(len(batch_image)):
                print(cos_sim(org_feat, vics_feat[i]),cos_sim(att_feat,vics_feat[j]))
            print('\n')
            # print(cos_sim(feat, vfeat), cos_sim(feat, vfeat1))
        with open('log/'+args.core+str(nowTime)+"_log.txt",'w') as fp2:
            for i, v in enumerate(loss_total):
                print(loss1[i], loss2[i], v, file=fp2)

        # visible = np.ones((112,112,3)) + 127
        image = [[0] for i in range(self.vis_long)]
        for i in range(self.vis_long):
            image[i] = diff + att_img[i+1]
            image[i] = image[i][:,:,::-1]
            image[i][image[i] < 0] = 0
            image[i][image[i] > 255] = 255
        # image = image.astype(np.uint8)
        # cv2.imshow("Image",image)
        # cv2.waitKey(0) == 'q'

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

    batch_image = []
    vic_images = []
    for idname in range(100):
        # str_idname = "%03d" % idname
        iddir = osp.join('picture', str(idname))
        filelist = os.listdir(iddir)
        att_image=os.path.join(iddir,filelist[0])
        att_img = cv2.imread(att_image)
        batch_image.append(att_img)
        vic_image = os.path.join(iddir, filelist[1])
        vic_img = cv2.imread(vic_image)
        vic_images.append(vic_img)


    # ta = datetime.datetime.now()
    diff,vis_diff = tool.generate(batch_image,vic_images, 0)

    # tb = datetime.datetime.now()
    # print( (tb-ta).total_seconds() )
    # save_name = '{}.png'.format(i)
    Time = datetime.datetime.now().strftime("%m-%d-%H-%M")
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