########################################################################################################################
# 测试脚本
########################################################################################################################

import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from osgeo import gdal
from src import unet_resnet50
import torch.nn.functional as F
import transforms as T


class GRID:            #读取坐标信息

    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # if 'int8' in img_data.dtype.name:
        #     datatype = gdal.GDT_Byte
        # elif 'int16' in img_data.dtype.name:
        #     datatype = gdal.GDT_UInt16
        # else:
        #     datatype = gdal.GDT_Float32
        datatype = gdal.GDT_Byte

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)


        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image

def detect_image(image):

    classes = 1                                                                                                         # 类别数
    weights_path = r"C:\Users\15089\Desktop\实验结果2\结果\SP_MPM_CA\ep050-loss0.024-val_loss0.119.pth"                       # 权重路径
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = unet_resnet50(num_classes=classes+1)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.to(device)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])

    image1 = np.transpose(image, (1, 2, 0))
    image2 = data_transform(image1)
    img = torch.unsqueeze(image2, dim=0)

    model.eval()
    with torch.no_grad():

        t_start = time.time()
        output = model(img.to(device))
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))
        output = output['out']

        pr = torch.squeeze(output, dim=0)
        pr = F.softmax(pr, dim=0).cpu().numpy()
        pr = pr.argmax(axis=0)

    masked_image = image.astype(np.uint32).copy()

    color = (180.0, 0.0, 0.0)                                                                                           # 自定义颜色



###########################################################
    for c in range(3):
        alpha = 1
        masked_image[c, :, :] = np.where(pr == 1,
                                         masked_image[c, :, :] *
                                         0.1 + alpha * color[c],
                                         masked_image[c, :, :]*0.5)
    # for c in range(3):
    #     alpha = 1
    #     masked_image[c, :, :] = np.where(pr == 1,
    #                                      masked_image[c, :, :] *
    #                                      (1 - alpha) + alpha * color[c],
    #                                      masked_image[c, :, :])
#####===========控制测试输出形式，当前是影像+预测区域，若只保留括号内为(pr == 1, 0 + alpha * color[c],0)，则只输出标签===========

    return masked_image

def main():

    IMAGE_DIR = r'C:\Users\15089\Desktop\qjz\test/'                      # 图像文件夹
    path_out = r"C:\Users\15089\Desktop\qjz\add/"                        # 输出文件夹

    run = GRID()
    count = os.listdir(IMAGE_DIR)
    for i in range(0, len(count)):
        path = os.path.join(IMAGE_DIR, count[i])

        proj, geotrans, data = run.load_image(path)
        r_image = detect_image(data)

        run.write_image(path_out + '{}.tif'.format(str(count[i])), proj, geotrans, r_image)



if __name__ == '__main__':
    main()
