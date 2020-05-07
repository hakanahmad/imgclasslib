import os
import numpy as np
import cv2
import pickle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from imgclasslib.model.alexnet import create_alexnet
from imgclasslib.model.googlenet import create_googlenet
from imgclasslib.model.inceptionv3 import create_inceptionv3
from imgclasslib.model.lenet import create_lenet
from imgclasslib.model.resnet50 import create_resnet50
from imgclasslib.model.squeezenet import create_squeezenet
from imgclasslib.model.vggnet import create_vggnet
from imgclasslib.model.darknet53 import create_darknet53
from imgclasslib.model.darknet19 import create_darknet19
from imgclasslib.model.inception_resnetv2 import create_inception_resnetv2
from imgclasslib.model.densenet201 import create_densenet201
from imgclasslib.model.densenet402 import create_densenet402

class ImageClassifier:
    def __init__(self):
        self.CATEGORIES = []
        self.IMG_SIZE = None
        self.DATADIR = None
        self.img_array = []
        self.new_array = []
        self.new_array1 = []
        self.test_array = []
        self.training_data = []
        self.testing_data = []
        self.X = []
        self.y = []
        self.save_pickle = None
        self.get_available_model = ['resnet50','lenet','alexnet','vggnet','googlenet','squeezenet','inceptionv3','darknet53','darknet19','inception_resnetv2','densenet201']
        self.model = None
        self.func = None
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []
        self.history = None
        self.predict = []
        self.result = []
        self.trained = False
        self.accuracy_list = []
        self.loss_list = []

    def createDataset(self,path,img_size,create_duplicate=True,save_pickle_file=False,split=0.1):
        '''
        This function is to create data from the path inserted. Please arrange your data like this:
        data
        -train
        --cat1
        (image)
        --cat2
        (image)
        --cat3
        (image)
        --...
        -test
        --cat1
        (image)
        --cat2
        (image)
        --cat3
        (image)
        --...
        img_size : (int) the size of the reshaped image.
        create_duplicate : (bool) This will create the double of the data that are in the folder, creating more training data as well.
        save_pickle_file : (bool) If true, the images and labels in the training data will be saved in .pickle file and can be loaded using loadFromPickle method.
        split : (float) [0,1] percentage of training data which are validation data.
        '''
        self.DATADIR = path
        self.IMG_SIZE = img_size
        path_train = os.path.join(self.DATADIR,'train')
        path_test = os.path.join(self.DATADIR,'test')
        self.CATEGORIES = [str(category) for category in os.listdir(path_train)]
        print('Creating Data...')
        for category in self.CATEGORIES:
            path_cat = os.path.join(path_train,category)
            self.class_num = self.CATEGORIES.index(category)
            desc = "training_data: {}".format(category)
            for img in tqdm(os.listdir(path_cat),ascii=True,desc=desc,ncols=100):
            #for img in os.listdir(path_cat):
                try:   
                    self.img_array = cv2.imread(os.path.join(path_cat,img))
                    self.img_array = cv2.cvtColor(self.img_array,cv2.COLOR_BGR2RGB)
                    self.new_array = cv2.resize(self.img_array,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([self.new_array,self.class_num])
                    if create_duplicate == True:
                        self.new_array1 = cv2.flip(self.new_array,1)
                        self.training_data.append([self.new_array1,self.class_num])
                    else:
                        pass
                except Exception as e:
                    pass
            #print('Training data class {} successfully created'.format(self.CATEGORIES[self.class_num]))
        #print('\nCreating Testing Data...')
        #for img in os.listdir(path_test):
        #    self.test_array = cv2.imread(os.path.join(path_test,img))
        #    self.test_array = cv2.resize(self.test_array,(self.IMG_SIZE,self.IMG_SIZE))
        #    self.testing_data.append(self.test_array)
        #self.testing_data = np.array(self.testing_data)
        for category in self.CATEGORIES:
            path_cat = os.path.join(path_test,category)
            self.class_num = self.CATEGORIES.index(category)
            desc = "testing_data: {}".format(category)
            for img in tqdm(os.listdir(path_cat),ascii=True,desc=desc,ncols=100):
            #for img in os.listdir(path_cat):
                try:   
                    self.test_array = cv2.imread(os.path.join(path_cat,img))
                    self.test_array = cv2.cvtColor(self.test_array,cv2.COLOR_BGR2RGB)
                    self.new_array = cv2.resize(self.test_array,(self.IMG_SIZE,self.IMG_SIZE))
                    self.testing_data.append([self.new_array,self.class_num])
                except Exception as e:
                    pass
            #print('Testing data class {} successfully created'.format(self.CATEGORIES[self.class_num]))
        print('Data successfully created')
        random.shuffle(self.training_data)
        random.shuffle(self.testing_data)
        print('Separating the images and labels in the training data...')
        self.X = []
        self.y = []
        for features,label in self.training_data:
            self.X.append(features)
            self.y.append(label)
        for features,label in self.testing_data:
            self.X_test.append(features)
            self.y_test.append(label)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        print('Separating complete...')
        if save_pickle_file == True:
            print('Saving to .pickle file')
            self.save_pickle=True
            pickle_out = open("X.pickle",'wb')
            pickle.dump(self.X,pickle_out)
            pickle_out.close()
            pickle_out = open("y.pickle",'wb')
            pickle.dump(self.y,pickle_out)
            pickle_out.close()    
            print('Saving complete...')
        if split>0 and split <1:
            self.X_train, self.y_train = self.X[:int(self.X.shape[0]*(1-split))], self.y[:int(self.X.shape[0]*(1-split))]
            self.X_val, self.y_val = self.X[int(self.X.shape[0]*(1-split)):], self.y[int(self.X.shape[0]*(1-split)):]
            print('All complete')
        else:
            print('Please enter the split arguement between 0 and 1')

        
    def loadFromPickle(self,X_pickle=None,y_pickle=None):
        '''
        X_pickle : (str) the path of the X.pickle file
        y_pickle : (str) the path of the y.pickle file
        '''
        if X_pickle != None and y_pickle != None:
            print('Loading data...')
            self.X = pickle.load(open(X_pickle,'rb'))
            self.y = pickle.load(open(y_pickle,'rb'))
            print('Loading complete')
        else:
            print('Please input the (X_pickle,y_pickle) argument')
        
    def setModel(self,model=None):
        '''
        model = (str) type of model
        '''
        if model == 'resnet50' and self.IMG_SIZE>=224:
            self.model = create_resnet50(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'lenet':
            self.model = create_lenet(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'alexnet' and self.IMG_SIZE>=224:
            self.model = create_alexnet(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'vggnet' and self.IMG_SIZE>=224:
            self.model = create_vggnet(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'googlenet' and self.IMG_SIZE>=224:
            self.model = create_googlenet(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'squeezenet' and self.IMG_SIZE >= 227:
            self.model = create_squeezenet(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'inceptionv3' and self.IMG_SIZE >= 299:
            self.model = create_inceptionv3(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'darknet53' and self.IMG_SIZE >= 256:
            self.model = create_darknet53(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'darknet19' and self.IMG_SIZE >= 256:
            self.model = create_darknet19(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'inception_resnetv2' and self.IMG_SIZE >= 299:
            self.model = create_inception_resnetv2(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'densenet201' and self.IMG_SIZE >= 224:
            self.model = create_densenet201(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        elif model == 'densenet402' and self.IMG_SIZE >= 224:
            self.model = create_densenet402(self.IMG_SIZE,num_categories=len(self.CATEGORIES))
        else:
            print("Cannot create model or the image size is wrong")
        if self.model != None:
            print('The model to use is {}'.format(str(model)))
            self.model.summary()
    
    def trainModel(self,batch_size=1,epochs=25):
        if self.model != None:
            print('Training Model...')
            self.history = self.model.fit(self.X_train,self.y_train,batch_size=batch_size,epochs=epochs,validation_data=(self.X_val,self.y_val),verbose=1)
            print('Finished Training Model')
            self.trained = True
        else:
            print("Model isn't defined")
    
    def evaluateModel(self):
        if self.trained:
            for his in list(self.history.history.keys()):
                if 'accuracy' in str(his):
                    self.accuracy_list.append(str(his))
                elif 'loss' in str(his):
                    self.loss_list.append(str(his))
            plt.subplot(1,2,1)
            for his in self.accuracy_list:
                plt.plot(self.history.history[str(his)])
            plt.title('Accuracy vs Validation Accuracy')
            plt.legend([str(his) for his in self.accuracy_list])
            plt.subplot(1,2,2)
            for his in self.loss_list:
                plt.plot(self.history.history[str(his)])
            plt.title('Loss vs Validation Loss')
            plt.legend([str(his) for his in self.loss_list])
            plt.tight_layout()
            plt.show()
        else:
            print("Model hasn't been trained")
    
    def predictTestData(self):
        if self.trained:
            #self.predict = np.array(self.model.predict(self.testing_data))
            self.predict = np.array(self.model.predict(self.X_test))
            if len(self.predict.shape) == 3:
                self.predict = self.predict[self.predict.shape[0]-1,:,:]
            plt.figure(figsize=(10,4*(len(self.testing_data)//2)))
            for pred in self.predict:
                self.result.append(self.CATEGORIES[np.argmax(pred)])
            print(self.result)
            for idx,im in enumerate(self.X_test):
                plt.subplot(len(self.testing_data),2,idx+1)
                plt.imshow(im)
                plt.title('Predicted: {} \nActual: {}'.format(str(self.result[idx]),str(self.CATEGORIES[self.y_test[idx]])))
                plt.axis('off')
            plt.tight_layout()
        else:
            print("Cannot predict data")
        plt.show()
