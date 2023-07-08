# FRUA
该项目实现跨图像攻击人脸识别模型的核心代码

介绍：

​	人脸识别技术的恶意运用可能会导致个人信息泄露，对个人隐私安全构成巨大威胁。因此，急需一种方法来遏制人脸识别系统非法访问身份信息。针对这一问题，通常采用为人脸图像设计针对性对抗样本攻击人脸识别系统的方法来保护个人隐私，但对于海量社交媒体图像，成本是高昂的。为此，提出了一种基于特征嵌入的通用人脸识别保护水印生成方法，通过对人脸图像对抗扰动共性计算，结合针对人脸识别模型特性设计的特征损失，生成基于特征扰动的保护水印。该水印可在不显著影响原始图像视觉质量的情况下直接叠加至人脸图像，无需进行额外计算，从而保护个人隐私免受未经授权的人脸识别系统的访问。

​	实验使用的人脸识别模型：Arcface(主干网络：MobileFace)、Arcface(主干网络：Resnet-IR)、SphereFace

​	实验使用的数据集：LFW、AgeDB、CFP-FD