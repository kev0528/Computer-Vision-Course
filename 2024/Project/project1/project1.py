import os
import numpy as np
import cv2

# RGB to YIQ Conversion Function
def rgb_to_yiq(image):
    yiq_matrix=np.array([[0.299, 0.587, 0.114],
                         [0.596,-0.274,-0.322],
                         [0.211,-0.523, 0.312]])
    return image @ yiq_matrix.T

# Histogram Feature Extraction 
def histogram(image):
    yiq_image=rgb_to_yiq(image)
    y_channel=yiq_image[:,:,0]
    hist=cv2.calcHist([y_channel.astype(np.uint8)],[0],None,[256],[0,256])
    return hist.flatten()

# Harris Corner Detection 
def harris_corner_detection(image):
    yiq_image=rgb_to_yiq(image)
    y_channel=yiq_image[:,:,0].astype(np.uint8)
    corners=cv2.cornerHarris(y_channel,2,3,0.04)
    return corners.flatten()[:425]

# SIFT Feature Extraction
def sift_feature(image):
    yiq_image=rgb_to_yiq(image)
    y_channel=yiq_image[:,:,0].astype(np.uint8)
    sift=cv2.SIFT_create()
    keypoints,descriptors=sift.detectAndCompute(y_channel,None)
    
    if descriptors is None:
        return np.zeros(128)
    
    feature_vector=descriptors.mean(axis=0)
    return feature_vector.flatten()

# SAD
def calculate_sad(test_image,train_image):
    return np.sum(np.abs(test_image-train_image))

# Nearest Neighbor
def find_nearest_neighbor(test_image,train_features,train_labels):
    nearest_distance=float('inf') # ∞
    nearest_label=None

    for i in range(len(train_features)):
        distance=calculate_sad(test_image,train_features[i])
        if distance<nearest_distance:
            nearest_distance=distance
            nearest_label=train_labels[i]

    return nearest_label
##############################################################################################################
# 影像路徑
train_folder_path='./data/train'
test_folder_path='./data/test'

# 特徵提取與訓練數據處理(train)
def load_train_data(feature_extraction_func):
    train_features=[]
    train_labels=[]
    for label_folder in os.listdir(train_folder_path):
        label_folder_path=os.path.join(train_folder_path,label_folder)
        if os.path.isdir(label_folder_path):
            for image_name in os.listdir(label_folder_path):
                image_path=os.path.join(label_folder_path,image_name)
                image=cv2.imread(image_path)
                if image is not None:
                    features=feature_extraction_func(image) 
                    train_features.append(features)
                    train_labels.append(label_folder)
    return train_features,train_labels

# 特徵提取與測試數據處理(test)
def load_test_data(feature_extraction_func):
    test_features=[]
    test_labels=[]
    for label_folder in os.listdir(test_folder_path):  
        label_folder_path=os.path.join(test_folder_path,label_folder)
        if os.path.isdir(label_folder_path):  
            for image_name in os.listdir(label_folder_path):
                image_path=os.path.join(label_folder_path,image_name)
                image=cv2.imread(image_path)  
                if image is not None:
                    features=feature_extraction_func(image) 
                    test_features.append(features)
                    test_labels.append(label_folder)  
    return test_features,test_labels

# 將測試影像進行分類並計算準確度
def classify_and_calculate_accuracy(train_features,train_labels,test_features,test_labels):
    correct_count=0 

    for i,test_feature in enumerate(test_features):
        # 使用最近鄰分類器找到預測標籤
        predicted_label=find_nearest_neighbor(test_feature,train_features,train_labels)
        
        if predicted_label==test_labels[i]:
            correct_count+=1 

    accuracy=correct_count/len(test_labels)
    return accuracy

train_features,train_labels=load_train_data(histogram)
test_features,test_labels=load_test_data(histogram)
accuracy=classify_and_calculate_accuracy(train_features,train_labels,test_features,test_labels)
print("Histogram: {:.2f}%".format(accuracy*100))

train_features,train_labels=load_train_data(harris_corner_detection)
test_features,test_labels=load_test_data(harris_corner_detection)
accuracy=classify_and_calculate_accuracy(train_features,train_labels,test_features,test_labels)
print("Harris Corner Detection: {:.2f}%".format(accuracy*100))

train_features,train_labels=load_train_data(sift_feature)
test_features,test_labels=load_test_data(sift_feature)
accuracy=classify_and_calculate_accuracy(train_features,train_labels,test_features,test_labels)
print("SIFT Feature: {:.2f}%".format(accuracy*100))