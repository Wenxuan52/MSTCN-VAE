import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import PIL.Image as Image

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_path = '../../autodl-tmp/iMiGUE/imigue_skeleton_train_viewsample'
Path = '../../autodl-tmp/iMiGUE/imigue_skeleton_train/'



for SampleID in os.listdir(Path):
    avi_path = video_path + '/' + 'Sample' + SampleID + '.avi'
    videoWriter = cv2.VideoWriter(avi_path, fourcc, fps, (640, 480))
    SamplePath = Path + SampleID + '/' + SampleID + '_light_hand.csv'
    print(SamplePath)
    df = pd.read_csv(SamplePath, header=None)
    coordinate = {}
    used_joints = ['Nose', 'ShoulderCenter', 'ShoulderLeft', 'ElbowLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight',
                   'HandRight', 'HipCenter', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'HipRight', 'KneeRight', 'AnkleRight',
                   'EyeLeft', 'EyeRight', 'EarLeft', 'EarRight', 'ToeLeft', 'SmallToeLeft', 'HeelLeft', 'ToeRight',
                   'SmallToeRight', 'HeelRight',
                   'RightFinger1', 'RightFinger2', 'RightFinger3', 'RightFinger4', 'RightFinger5',
                   'LeftFinger1', 'LeftFinger2', 'LeftFinger3', 'LeftFinger4', 'LeftFinger5']

    SkeletonConnectionMap = (
        ['ShoulderCenter', 'Nose'], ['ShoulderCenter', 'ShoulderLeft'], ['ShoulderLeft', 'ElbowLeft'],
        ['ElbowLeft', 'HandLeft'],
        ['ShoulderCenter', 'ShoulderRight'], ['ShoulderRight', 'ElbowRight'], ['ElbowRight', 'HandRight'],
        ['Nose', 'EyeLeft'], ['EyeLeft', 'EarLeft'],
        ['Nose', 'EyeRight'], ['EyeRight', 'EarRight'], ['HandLeft', 'LeftFinger1'], ['HandLeft', 'LeftFinger2'],
        ['HandLeft', 'LeftFinger3'], ['HandLeft', 'LeftFinger4'],
        ['HandLeft', 'LeftFinger5'], ['HandRight', 'RightFinger1'], ['HandRight', 'RightFinger2'],
        ['HandRight', 'RightFinger3'], ['HandRight', 'RightFinger4'],
        ['HandRight', 'RightFinger5'])

    for index, row in df.iterrows():
        if max(row[1:]) > 0:
            pos = 1
            x = []
            y = []
            coordinate[index] = {}
            existpoints = {}
            for P_feature in used_joints:
                coordinate[index][P_feature] = row[pos:pos + 3]
                if coordinate[index][P_feature][pos] == 0 and coordinate[index][P_feature][pos + 1] == 0:
                    pos += 3
                else:
                    existpoints[P_feature] = (coordinate[index][P_feature][pos], -1 * coordinate[index][P_feature][pos + 1])
                    x.append(coordinate[index][P_feature][pos])
                    y.append(-1 * coordinate[index][P_feature][pos + 1])
                    pos += 3
            
            if index % 100 == 0:
                print('Processing ' + ' Sample ' + SampleID + ' frame ' + str(index))
            
            plt.scatter(x, y)
            plt.xlim((0, 1200))
            plt.ylim((-1200, 0))
            for pointsonline in SkeletonConnectionMap:
                if pointsonline[0] in existpoints and pointsonline[1] in existpoints:
                    x_num = [existpoints[pointsonline[0]][0], existpoints[pointsonline[1]][0]]
                    y_num = [existpoints[pointsonline[0]][1], existpoints[pointsonline[1]][1]]
                    plt.plot(x_num, y_num, color='r')
            # 4.保存图片
            canvas = FigureCanvasAgg(plt.gcf())
            # 绘制图像
            canvas.draw()
            # 获取图像尺寸
            w, h = canvas.get_width_height()
            # 解码string 得到argb图像
            buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            # 重构成w h 4(argb)图像
            buf.shape = (w, h, 4)
            # 转换为 RGBA
            buf = np.roll(buf, 3, axis=2)
            # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
            image = Image.frombytes("RGBA", (w, h), buf.tobytes())
            # 转换为numpy array rgba四通道数组
            image = np.asarray(image)
            # 转换为rgb图像
            rgb_image = image[:, :, :3]
            # 转换为bgr图像
            r, g, b = cv2.split(rgb_image)
            img_bgr = cv2.merge([b, g, r])
            '''生成视频'''
            videoWriter.write(img_bgr)
            '''释放内存'''
            plt.clf()
            plt.close()
