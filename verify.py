import tqdm
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master")
from config import get_config
from Learner import face_learner
from utils_insight import load_facebank, draw_box_name, prepare_facebank

import argparse
from scrfd import SCRFD
from utils import norm_crop, logSaver
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch


if __name__ == '__main__':
    with logSaver('error.txt'):
        parser = argparse.ArgumentParser(description='for face verification')
        parser.add_argument("-s", "--save", help="whether save", action="store_true")
        parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
        parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
        parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
        parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
        args = parser.parse_args()

        sys.path.append("/data/hdd/duanwei/FRUA/sphereface_pytorch-master")
        import net_sphere
        sys.path.append("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master/")
        from config import get_config

        conf = get_config(use_mobilfacenet=True)
        # net = MobileFaceNet(conf.embedding_size).to(conf.device)
        # net.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/InsightFace_Pytorch-master/work_space/models/model_mobilefacenet.pth",
        #                map_location=conf.device))

        net = getattr(net_sphere, 'sphere20a')()
        net.load_state_dict(torch.load('/data/hdd/duanwei/FRUA/sphereface_pytorch-master/model/sphere20a_20171020.pth'))
        net.feature = True
        #
        # conf = get_config(use_mobilfacenet=False)
        # net = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        # net.load_state_dict(torch.load("/data/hdd/duanwei/FRUA/InsightFace/work_space/models/model_ir_se50.pth",
        #                                map_location=conf.device))
        net.eval().to(conf.device)

        #prepare detctor
        detector = SCRFD(model_file="/data/hdd/duanwei/FRUA/some-resources-master/assets/det_10g.onnx")
        ctx_id = -1 if not True else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        print('SCRFD loaded')

        #prepare_model
        conf = get_config(False)
        learner = face_learner(conf, True)
        learner.threshold = args.threshold
        # if conf.device.type == 'cpu':
        #     learner.load_state(conf, 'cpu_final.pth', True, True)
        # else:
        #     learner.load_state(conf, 'ir_se50.pth', False, True)
        #
        # learner.model.eval()
        print('learner loaded')

        if args.update:
            targets, names = prepare_facebank(conf, net, detector, tta=args.tta)
            print('facebank updated')
        else:
            targets, names = load_facebank(conf)
            print('facebank loaded')

        vaild_num = []
        error_num = 0
        img_lfw = []
        # root_path = "/data/hdd/duanwei/FRUA/data/lfw"
        # lst = os.listdir(root_path)
        class_num = 0
        file = np.load("dataset/feature_age_std.npy")


        def getFileList(dir, Filelist, ext=None):
            """
            获取文件夹及其子文件夹中文件列表
            输入 dir：文件夹根目录
            输入 ext: 扩展名
            返回： 文件路径列表
            """
            newDir = dir
            if os.path.isfile(dir):
                if ext is None:
                    Filelist.append(dir)
                else:
                    if ext in dir[-3:]:  # jpg为-3/py为-2
                        Filelist.append(dir)

            elif os.path.isdir(dir):
                for s in os.listdir(dir):
                    newDir = os.path.join(dir, s)
                    getFileList(newDir, Filelist, ext)
            return Filelist
        # dict1 = {}
        dict2  = {}
        total = 0
        iter = 0
        pathls = "/data/hdd/duanwei/dataset/cfpbase112*96"
        imglist = getFileList(pathls,[],'jpg')
        for img in imglist:
            image = cv2.imread(img)
            image1 = cv2.resize(image,(112,96))
            cv2.imwrite(img,image1)

        # for path in pathls:
        #
        #     # imglist = getFileList("/data/hdd/duanwei/dataset/cfp-dataset/Data/Images", [], 'jpg')
        #     imglist = getFileList(os.path.join("/data/hdd/duanwei/dataset/cfp-dataset/Data/Images",path), [], 'jpg')
        #     len_name = len(imglist)
        #     # total += len_name
        #     if(len_name in range(5,26)):
        #         total += 1
        #         if (total == 221):
        #             sys.exit(0)
        #         if (path not in dict2.keys()):
        #             dict2[path] = iter
        #             iter += 1
        #         it = 0
        #         for imgpath in tqdm(imglist):
        #             os.makedirs("/data/hdd/duanwei/dataset/cfpbase/" + str(dict2[path]),exist_ok=True)
        #             # try:
        #             frame = cv2.imread(imgpath)
        #             bboxes, kpss = detector.detect(frame, max_num=1)
        #             if bboxes.shape[0] == 0:
        #                 continue
        #             org_img, M = norm_crop(frame, kpss[0], image_size=112)
        #             if (org_img.shape == (112,112,3)):
        #                 cv2.imwrite("/data/hdd/duanwei/dataset/cfpbase/" + str(dict2[path]) +"/"+str(it)+".jpg",org_img)
        #                 it +=1
        #     print(it)
                #     img_lfw.append(org_img[:, :, ::-1])
                # if len(img_lfw) == 100:
                #     break
            # except:
            #     continue
        # np.save("dataset/feature_age_std.npy", img_lfw)
        # for file in tqdm(lst):
        #     path_1 = os.path.join(root_path, file)
        #     class_num += 1
        #     if os.path.isdir(path_1):
        #         path_2 = os.listdir(path_1)
        #         for img in path_2:
        #             # try:
        #                 frame = cv2.imread(os.path.join(path_1, img))
        #                 # frame = cv2.imread("/data/hdd/duanwei/FRUA/data/stdface/5/304.jpg")
        #                 pert = np.load("/data/hdd/duanwei/FRUA/some-resources-master/output/03-12-09-55diff.npy")
        #                 # cv2.namedWindow('demo', 0)
        #                 # cv2.resizeWindow('demo', 600, 500)
        #                 # cv2.imshow('demo', frame_new)
        #                 # cv2.waitKey(0)
        #                 # cv2.destroyAllWindows()
        #                 # image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) #bgr to rgb
        #                 # frame = frame + pert[:,:,::-1]
        #                 # cv2.imwrite("1.jpg",frame)
        #                 bboxes, kpss = detector.detect(frame, max_num=1)
        #                 if bboxes.shape[0] == 0:
        #                     continue
        #                 org_img, M = norm_crop(frame, kpss[0], image_size=112)
        #                 img_lfw.append(org_img[:, :, ::-1])
        #                 # face = np.array(org_img)
        #                 # # face = face + cv2.resize(pert,(112,112))
        #                 # # face = face + pert
        #                 # face = face + (20 * np.random.random((112,112,3)) -10)
        #                 # face = np.round(face)
        #                 # face = face.astype(np.float32)
        #                 # face = np.clip(face, 0, 255)
        #
        #                 # cv2.imwrite("1.png",face,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        #                 # face_1 = cv2.imread("1.png")
        #
        #                 # results, score = learner.infer(conf, face, targets, args.tta)
        #                 # if names[results[0] + 1] == file:
        #                 #     vaild_num.append(1)
        #                 # else:
        #                 #     vaild_num.append(0)
        #             # except:
        #             #     error_num += 1
        #     else:
        #         continue
        # np.save("feature_lfw.npy",img_lfw)
        # att_rat = sum(vaild_num) / len(vaild_num)
        # # print("after round clip attack_ratio")
        # print("orgin_sucess")
        # print(att_rat)
        #
        # print(len(vaild_num))
        # print(error_num)
        # summary = " ArcFace(resnet50) + LFW +pert(make for SphereFace(112*96))_sucess"
        # with open("log.txt", 'a') as fp2:
        #     print('summary %s' % (summary), file=fp2)
        #     print('sus %f' % (att_rat), file=fp2)
        #     print('total image %d' % (len(vaild_num)), file=fp2)
        #     print('can not det image %d' % (error_num), file=fp2)
        #     print("-----------------------------\n",file = fp2)
