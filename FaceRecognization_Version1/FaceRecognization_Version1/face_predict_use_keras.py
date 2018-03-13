# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:43:14 2018

@author: yuwangwang
"""

"""============================================================================
   2018.2.17版本一，单人单目标判断
   1、对输入的摄像头图像进行预测，判断类别
   
   2018.2.19版本二，多人单目标判断
"""
  
import cv2
import sys
import gc
from face_train_use_keras import Model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    # 加载模型
    model = Model()
    model.load_model(file_path = '/home/yuwangwang/FaceRecognization_Version1/model/me.face.model.h5')    
              
    # 框住人脸的矩形边框颜色,颜色为绿色      
    color = (0, 255, 0)
    
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(int(sys.argv[1]))
    
    # 人脸识别分类器本地存储路径
    cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"    
    
    # 循环检测识别人脸
    while True:
        # 读取一帧视频
        _, frame = cap.read()  
        
        # 图像灰度化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        # 利用分类器识别出哪个区域为人脸
        # 参数1：image--待检测图片，一般为灰度图像加快检测速度；
        # 参数2：objects--被检测物体的矩形框向量组；
        # 参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
        # 参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        # 如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        # 如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        # 这种设定值一般用在用户自定义对检测结果的组合程序上；
        # 参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
        # CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
        # 因此这些区域通常不会是人脸所在区域；
        # 参数6、7：minSize和maxSize用来限制得到的目标区域的范围。
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                
                # 如果是“我”
                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    # 文字提示是谁，颜色为粉色
                    cv2.putText(frame,'Me', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    pass
                            
        cv2.imshow("识别我", frame)
        
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()