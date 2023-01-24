import os
from glob import glob # extract path of each file
import pandas as pd # data preprocessing
from xml.etree import ElementTree as et # parse information from XML
from functools import reduce
from shutil import move
import warnings
warnings.filterwarnings('ignore')
import sys
from source.exception import ObjectDetectionException
from source.logger import logging

class Data_Preprocess:
    def __init__():
        pass

    def preprocess():
        '''preprocess all data'''
        try:

            # get path of each xml file
            xmlfiles = glob('source/data_images/*.xml')
            # replace \\ with /
            replace_text = lambda x: x.replace('\\','/')
            xmlfiles = list(map(replace_text,xmlfiles))

            # read xml files
            # from each xml file we need to extract
            # filename, size(width, height), object(name, xmin, xmax, ymin, ymax)
            def extract_text(filename):
                tree = et.parse(filename)
                root = tree.getroot()

                # extract filename
                image_name = root.find('filename').text
                # width and height of the image
                width = root.find('size').find('width').text
                height = root.find('size').find('height').text
                objs = root.findall('object')
                parser = []
                for obj in objs:
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = bndbox.find('xmin').text
                    xmax = bndbox.find('xmax').text
                    ymin = bndbox.find('ymin').text
                    ymax = bndbox.find('ymax').text
                    parser.append([image_name, width, height, name,xmin,xmax,ymin,ymax])
                    
                return parser

            parser_all = list(map(extract_text,xmlfiles))
            data = reduce(lambda x, y : x+y,parser_all)
            df = pd.DataFrame(data,columns = ['filename','width','height','name','xmin','xmax','ymin','ymax']) # convert into pandas dataframe

            # datatype conversion
            cols = ['width','height','xmin','xmax','ymin','ymax']
            df[cols] = df[cols].astype(int)

            # calculate for center x, center y each bounding box.
            df['center_x'] = ((df['xmax']+df['xmin'])/2)/df['width']
            df['center_y'] = ((df['ymax']+df['ymin'])/2)/df['height']
            # w 
            df['w'] = (df['xmax']-df['xmin'])/df['width']
            # h 
            df['h'] = (df['ymax']-df['ymin'])/df['height']

            # split data into train and test
            # 80% train and 20% test
            images = df['filename'].unique()
            img_df = pd.DataFrame(images,columns=['filename'])
            img_train = tuple(img_df.sample(frac=0.8)['filename']) # shuffle and pick 80% of images
            img_test = tuple(img_df.query(f'filename not in {img_train}')['filename']) # take rest 20% images

            train_df = df.query(f'filename in {img_train}')
            test_df = df.query(f'filename in {img_test}')

            #Assign id number to object names
            # label encoding
            def label_encoding(x):
                labels = {'person':0, 'car':1, 'chair':2, 'bottle':3, 'pottedplant':4, 'bird':5, 'dog':6,
                'sofa':7, 'bicycle':8, 'horse':9, 'boat':10, 'motorbike':11, 'cat':12, 'tvmonitor':13,
                'cow':14, 'sheep':15, 'aeroplane':16, 'train':17, 'diningtable':18, 'bus':19}
                return labels[x]

            train_df['id'] = train_df['name'].apply(label_encoding)
            test_df['id'] = test_df['name'].apply(label_encoding)

            # Save Image and Labels in text
            train_folder = 'source/model_training/yolov5/src/data_images/train'
            test_folder = 'source/model_training/yolov5/src/data_images/test'

            os.mkdir(train_folder)
            os.mkdir(test_folder)

            # groupby with filename, so we will get information of all bounding boxes for that file/image in single file. 
            cols = ['filename','id','center_x','center_y', 'w', 'h']
            groupby_obj_train = train_df[cols].groupby('filename')
            groupby_obj_test = test_df[cols].groupby('filename')

            #groupby_obj_train.get_group('000009.jpg').set_index('filename').to_csv('sample.txt',index=False,header=False)
            # save each image in train/test folder and repective labels in .txt
            def save_data(filename, folder_path, group_obj):
                # move image
                src = os.path.join('source/data_images',filename)
                dst = os.path.join(folder_path,filename)
                move(src,dst) # move image to the destination folder
                
                # save the labels
                text_filename = os.path.join(folder_path,
                                            os.path.splitext(filename)[0]+'.txt')
                group_obj.get_group(filename).set_index('filename').to_csv(text_filename,sep=' ',index=False,header=False)

            filename_series = pd.Series(groupby_obj_train.groups.keys())
            filename_series.apply(save_data,args=(train_folder,groupby_obj_train))

            filename_series_test = pd.Series(groupby_obj_test.groups.keys())
            filename_series_test.apply(save_data,args=(test_folder,groupby_obj_test))
            logging.info("preprocessing completed")
        
        except Exception as e:
                raise  ObjectDetectionException(e,sys)




Data_Preprocess.preprocess()