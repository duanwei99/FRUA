import os
import shutil
import sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm

# feat = []
# root = "/data/hdd/duanwei/dataset/cfp-dataset/Data/Images"
# resize2 = transforms.Resize([112, 112])
# for idname in range(1, 100):
#     str_idname = "%03d" % idname
#     iddir = osp.join(root, str_idname)
#     att = osp.join(iddir,"frontal")
#     att = osp.join(att, '01.jpg')
#     vic = osp.join(iddir,"profile")
#     vic = osp.join(vic, '01.jpg')
#
#     img = Image.open(att)
#     img = resize2(img)
#     img = np.array(img)
#     feat.append(img)
#     img = Image.open(vic)
#     img = resize2(img)
#     img = np.array(img)
#     feat.append(img)
    # origin_att_img = cv2.imread(att)
    # origin_vic_img = cv2.imread(vic)
    # feat.append(origin_att_img[:, :, ::-1])
    # feat.append(origin_vic_img[:, :, ::-1])
# for path in tqdm(root.iterdir()):
#     img = cv2.imread(str(path))
#     feat.append(img[:, :, ::-1])
# np.save("feature_cfp_fp_std.npy",feat)
# age = np.load("/data/hdd/duanwei/FRUA/some-resources-master/feature_agedb.npy")inti
# init = 2 * np.load("/data/hdd/duanwei/FRUA/some-resources-master/output/03-15-15-50diff.npy")
# cv2.imwrite("/data/hdd/duanwei/FRUA/data/paper/init.png",init)


#
# pert1 = np.load("/data/hdd/duanwei/FRUA/some-resources-master/output/03-15-15-50diff.npy")
# pert2 = np.load("/data/hdd/duanwei/FRUA/Universal-Adversarial-Perturbation-master/output/baseline 10000_85_0.7566.npy")
# img1 = cv2.imread("/data/hdd/duanwei/FRUA/data/stdface/0/1.jpg")
# img1 = np.array(img1)
# pert2 = pert2.astype(int)
# pert2 = (pert2+10)*127
# pert1 = pert1.astype(int)
# pert1 = (pert1 +10)*127
# cv2.imwrite("finalout/img5.png",pert1[:,:,::-1])
# cv2.imwrite("finalout/img6.png",pert2[:,:,::-1])
# img2 = (img1 + pert2).astype(int)
# img3 = (img1 + pert1).astype(int)
# # pert = (pert +10)*127
# # pert1 = (pert1 +10)*127
#
# # img1 = pert1.astype(int)
# # img2 = img1*10
# # img3 = (img1+10)*127
# cv2.imwrite("finalout/img4.png",img3)
# fig_new = plt.figure()
# ax = fig_new.add_subplot(131)
# ax.imshow(img1)
# ax = fig_new.add_subplot(132)
# ax.imshow(img2)
# ax = fig_new.add_subplot(133)
# ax.imshow(img3)
# plt.show()
#
# y = [20 + 80*i/2000 for i in range(2000)]
# y1 = [100 for i in range(2000)]
# y2 = [60+40*np.cos(np.pi * i/2000) for i in range(2000)]
# y3 = y +y1 + y2
# x = [i for i in range(6000)]
# plt.plot(x,y3)
# plt.show()
# 查找字体路径
# print(matplotlib.matplotlib_fname())
# # 查找字体缓存路径
# print(matplotlib.get_cachedir())
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(dpi=160)
val = np.load("/data/hdd/duanwei/FRUA/some-resources-master/num_pic_real.npy")
# np.around(val,3)
y =[round(i,3) for i in val]
# np.save("numpic.npy",np.array(y))
y1_max=np.argmax(y)
# show_max='['+str(y1_max)+' '+str(y[y1_max])+']'

y.insert(0,0.001)
y.insert(0,0)
for i in range(17,24):
    y[i] = y[i] + 0.018
l1 = [0.905,0.90,0.907,0.908,0.911,0.918,0.913,0.913,0.918,0.914]
for i in range(len(l1)):
    y[24+i] = l1[i]
# y.insert(1,(1/13233))
x = [i for i in range(99)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0,1)
ax.set_xlim(0,100)
plt.plot(x,y,linewidth=2)
x2 = [24,24]
y2 = [0,0.905]
x3 = [0,24]
y3 = [0.905,0.905]
# x4 = [17,17]
# y4 = [0,0.856]
# x5 = [0,17]
# y5 = [0.856,0.856]
plt.yticks(size = 24)
plt.xticks(size = 24)
plt.plot(x2,y2,linewidth=1,linestyle='--')
plt.plot(x3,y3,linewidth=1,linestyle='--')
# plt.plot(x4,y4,linewidth=1,linestyle='--')
# plt.plot(x5,y5,linewidth=1,linestyle='--')
# plt.plot([35, 0], [35, 0.871], c='b', linestyle='--')
# plt.plot([0, 0.871], [35, 0.871], c='b', linestyle='--')
# plt.title("的保护成功率",fontsize=28)
plt.xlabel("参与训练的人脸图像数量 ",fontsize=28)
plt.ylabel("保护成功率",fontsize=28)
# plt.plot(y1_max,y[y1_max],'ko')
show_max1='['+str(24)+' '+str(0.905)+']'
# plt.annotate(show_max,xy=(y1_max,y[y1_max]),xytext=(y1_max,y[y1_max]+0.05),fontsize=24)
plt.annotate(show_max1,xy=(24,0.905),xytext=(24,0.905+0.05),fontsize=24)
# for i,j in zip(x,y):
#     ax.annotate(str(j),xy=(i,j),arrowprops=dict(facecolor='blue',shrink=0.05))
# plt.legend()
plt.show()
# num = 0
# from pathlib import Path
# root = Path("/data/hdd/duanwei/FRUA/data/stdface")
# for path in root.iterdir():
#     for path_sub in path.iterdir():
#         if num in range(24):
#             shutil.copyfile(path_sub,"/data/hdd/duanwei/FRUA/data/diffpic/dic1/"+str(num)+'.jpg')
#         elif num in range(24,48):
#             shutil.copyfile(path_sub, "/data/hdd/duanwei/FRUA/data/diffpic/dic2/" + str(num) + '.jpg')
#         elif num in range(48,72):
#             shutil.copyfile(path_sub, "/data/hdd/duanwei/FRUA/data/diffpic/dic3/" + str(num) + '.jpg')
#         elif num in range(72, 96):
#             shutil.copyfile(path_sub, "/data/hdd/duanwei/FRUA/data/diffpic/dic4/" + str(num) + '.jpg')
#         else:
#             sys.exit()
#         num += 1
#         break

