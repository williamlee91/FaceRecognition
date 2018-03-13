# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:09:40 2018

@author: yuwangwang
"""

"""============================================================================
   1、利用haar函数检测人脸
   2、检测到人脸后保存到指定文件夹
   3、数据集尽量多采集
"""

# 导入函数
import cv2
import sys
from PIL import Image

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    # 本次采集采用笔记本自带摄像头
    cap = cv2.VideoCapture(camera_idx)                
    
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
    
    # 识别出人脸后要画的边框的颜色，RGB格式，颜色为绿色
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        
        # ok可以用ret替代，因为可以用_替代
        ok, frame = cap.read() # 读取一帧数据
        if not ok:            
            break                
        # 将当前桢图像转换成灰度图像，可以降低计算复杂度
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)           
        
        # image—Mat类型的图像
        # objects—检测得到的矩形
        # rejectLevels—如果不符合特征的矩形，返回级联分类器中符合的强分类器数
        # scaleFactor—图像缩放因子，此处设置为1.2
        # minNeighbors—设置为3是为了消除误报，但是过大又会失去所有的人脸
        # minObjectSize—最小检测窗口大小，此处设置为32 × 32
        # maxObjectSize—最大检测窗口大小
        # outputRejectLevels—是否输出rejectLevels和levelWeights，默认为false
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          # 大于0则表示检测到人脸                                   
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect   # 获取识别信息的坐标信息                     
                
                # 将当前帧保存为图片并绘制方框
                img_name = '%s/%d.jpg'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   # 如果超过指定最大保存数量退出循环
                    break
                
                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                # 显示当前捕捉到了多少人脸图片
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): 
            break                
                       
        # 显示图像，等待10毫秒或者检测到按下"q",则退出
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
 
"""============================================================================
   采用后台终端输入命令，命令格式为python+代码文件名+cam_id+图片数量+保存全路径
"""   
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        # path_name一定要写全路径，否则找不到指定的路径，自然无法保存图片
        CatchPICFromVideo("截取人脸", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])