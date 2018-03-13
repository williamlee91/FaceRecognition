# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:32:08 2018

@author: yuwangwang

@contact: QQ:632207491 E-mail:yuwangwang91@163.com
"""

"""============================================================================
   2018.2.16版本一
   训练一个实用的模型，暂时先做2分类
   1、对导入的数据分成三类
   2、设计一个CNN网络并进行训练保存
   
   2018.2.19版本二
   1、训练一个实用的模型，做多分类，目标到5类别分类
"""
# random() 方法返回随机生成的一个实数，它在[0,1)范围内
import random
import numpy as np
# 为了保存最后的h5文件
import h5py
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
# 对load_face_dataset的数据集进行导入
from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

"""============================================================================
   1、按照交叉验证的原则将数据集划分成三部分：训练集、验证集、测试集；
   2、按照keras库运行的后端系统要求改变图像数据的维度顺序；
   3、将数据标签进行one-hot编码，使其向量化
   4、归一化图像数据         
"""
class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        
        # 测试集
        self.test_images  = None            
        self.test_labels  = None
        
        # 数据集加载路径
        self.path_name    = path_name
        
        # 当前库采用的维度顺序
        self.input_shape = None
        
    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 3, nb_classes = 2):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)        
        
        # 导入了sklearn库的交叉验证模块，利用函数train_test_split()来划分训练集和验证集
        # 划分出了30%的数据用于验证，70%用于训练模型
        train_images, valid_images, train_labels, valid_labels = train_test_split(images,\
        labels, test_size = 0.3, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5,\
        random_state = random.randint(0, 100))                
        
        # 当前的维度顺序如果为'channels_first'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            
            
            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            # 将其归一化,图像的各像素值归一化到0~1区间，数据集先浮点后归一化的目的是提升网络收敛速度，
            # 减少训练时间，同时适应值域在（0,1）之间的激活函数，增大区分度
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels

"""============================================================================            
    1、采用CNN传统的sequential网络结构 
    2、包含4个卷基层，2个maxpooling，2个FC层
    3、采用的激活函数是"Relu",最后采用softmax做分类
    4、nb_classes为两个分类标签        
"""           
class Model:
    def __init__(self):
        self.model = None 
        
    # 建立模型
    def build_model(self, dataset, nb_classes = 2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential() 
        
        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        
        #1 2维卷积层1，卷积核为32 × 3 × 3,边界为"same"就是保持卷积后图像大小不变，就是需要padding填充
        #  输入图形一般就是数据集的形状，然后输出为64 × 64 × 32，参数为（3 × 3 × 3 + 1） × 32 = 896
        self.model.add(Conv2D(32, (3, 3), padding='same', 
                                     input_shape = dataset.input_shape))    
        #2 激活函数层，输出为64 × 64 × 32        
        self.model.add(Activation('relu'))                                 
        
        #3 2维卷积层2，卷积核为32 × 3 × 3,输出为62 × 62 × 32，参数为9248
        self.model.add(Conv2D(32, (3, 3)))                                                       
        
        #4 激活函数层，输出为62 × 62 × 32 
        self.model.add(Activation('relu'))                                  
        
        #5 池化层1，池化区域为2 × 2，输出为31 × 31 × 32
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      
        
        #6 Dropout层1，dropout率为0.25，输出为31 × 31 × 32
        self.model.add(Dropout(0.25))                                       
        
        #7 2维卷积层3，卷积核为64 × 3 × 3，边界为"same"就是保持卷积后图像大小不变，就是需要padding填充
        #  输出为31 × 31 × 64，参数为18496
        self.model.add(Conv2D(64, (3, 3), padding='same'))         
       
        #8  激活函数层，输出为31 × 31 × 64
        self.model.add(Activation('relu'))                                  
        
        #9  2维卷积层4，卷积核为32 × 3 × 3,输出为29 × 29 × 64，参数为36928
        self.model.add(Conv2D(64, (3, 3)))                             
        
        #10 激活函数层，输出为输出为29 × 29 × 64
        self.model.add(Activation('relu'))                                  
        
        #11 池化层2，池化区域为2 × 2，输出为14 × 14 × 64
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      
        
        #12 Dropout层2，dropout率为0.25，输出为14 × 14 × 64
        self.model.add(Dropout(0.25))                                       
        
        #13 Flatten层，输出14 × 14 × 64=12455
        self.model.add(Flatten())  

        #14 Dense层1,又被称作全连接层，输出为512，参数为6423040                                 
        self.model.add(Dense(512))                                          
        
        #15 激活函数层，输出为512   
        self.model.add(Activation('relu')) 

        #16 Dropout层3，dropout率设置为0.5                        
        self.model.add(Dropout(0.5))                                        
        
        #17 Dense层2，输出为2，参数为1026
        self.model.add(Dense(nb_classes))                                   
        
        #18 分类层，输出最终结果
        self.model.add(Activation('softmax'))                               
        
        #输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size = 20, epochs = 5, data_augmentation = True):        
        
        # 采用SGD + momentum的优化器进行训练，首先生成一个优化器对象
        # momentum指定动量值，让优化器在一定程度上保留之前的优化方向，同时利用当前样本微调最终的
        # 优化方向，这样即能增加稳定性，提高学习速度，又在一定程度上避免了陷入局部最优陷阱
        # 参数其值为0~1之间的浮点数。一般来说，选择一个在0.5 ~ 0.9之间的数即可
        # 代码中SGD函数的最后一个参数nesterov则用于指定是否采用nesterov动量方法
        # nesterov momentum是对传统动量法的一个改进方法，其效率更高
        # 并完成实际的模型配置工作
        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True) 
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,
                           metrics=['accuracy'])   
        
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           epochs = epochs,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        # 使用实时数据提升
        else:            
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             # 是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  # 是否对输入数据施以ZCA白化
                rotation_range = 20,                    # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               # 同上，只不过这里是垂直
                horizontal_flip = True,                 # 是否进行随机水平翻转
                vertical_flip = False)                  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)                        

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     steps_per_epoch = dataset.train_images.shape[0],
                                     epochs = epochs,

                                     validation_data = (dataset.valid_images, dataset.valid_labels))

    # 一个函数用于保存模型，一个函数用于加载模型。
    # keras库利用了压缩效率更高的HDF5保存模型，所以我们用“.h5”作为文件后缀                                 
    MODEL_PATH = '/home/yuwangwang/FaceRecognization_Version1/model/me.face.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)
    

    # 模型评估   
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))    
    # 识别人脸
    def face_predict(self, image):    
        # 依然是根据后端系统确定维度顺序
        # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                            
            # 与模型训练不同，这次只是针对1张图片进行预测    
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))    
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    
        
        # 浮点并归一化
        image = image.astype('float32')
        image /= 255
        
        # 给出输入属于各个类别的概率，我们是二值类别
        # 则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)
        
        # 给出类别预测结果为0或者1
        result = self.model.predict_classes(image)        

        # 返回类别预测结果
        return result[0]

"""============================================================================
   1、训练模型和评估模型只能使用一个，顺序采用先训练模型，在进行评估模型
   2、训练玩后的模型保存名为me.face.model.h5
   3、保存路径为model文件下
"""


if __name__ == '__main__':
    dataset = Dataset('/home/yuwangwang/FaceRecognization_Version1/face_data')    
    dataset.load()
    
    """
    #训练模型
    model = Model()
    #先前添加的测试build_model()函数的代码
    model.build_model(dataset)
    #测试训练函数的代码
    model.train(dataset)
    model.save_model(file_path = '/home/yuwangwang/FaceRecognization_Version1/model/me.face.model.h5')
    """
    #评估模型
    model = Model()
    model.load_model(file_path = '/home/yuwangwang/FaceRecognization_Version1/model/me.face.model.h5')
    model.evaluate(dataset)
