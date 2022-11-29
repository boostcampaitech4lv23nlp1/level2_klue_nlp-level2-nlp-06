## TODO: Move some Preprocessing function to this file.

import torch
import pandas as pd
from sklearn.model_selection import train_test_split


# Get weight from label distribution for 'weighted CrossEntropy'
def get_weights_prob(label2num: dict, df: pd.DataFrame, weighted=0):
    '''
    Computes weight distribution of data with Label num for CrossEntropy Loss.
    
    Args:
        label2num : Dictionary which maps label to integer.
        df : 'Trainset' DataFrame
        weighted : 'True' if you want to use weighted CrossEntropy. 
                    It'll return uniform distirbution if you set it to False.
    '''
    if weighted == 1:
        weights = []

        no_relation_index = 0
        for i, label in enumerate(label2num.keys()):
            if label == "no_relation":
                no_relation_index = i
                continue
            cnt = len(df[df["label"]==label])
            weights.append(cnt)
            
        # 제곱으로 라벨 비율에 따른 차이를 더 늘려보자.
        weights = [(1 - (weight/sum(weights)))**2 for weight in weights]
        # no_relation에 더 큰 패널티를 부여. (0.5)
        weights = weights[:no_relation_index] + [0.5] + weights[no_relation_index:]
    else:
        weights = [1 for _ in range(len(label2num.keys()))]
        
    return torch.Tensor(weights)


def preprocessing_dataset(data: pd.DataFrame):
    """
    initial dataset 내부의 entity를 사용하기 좋게 변형해줍니다.

    Args:
        data (DataFrame): 전처리를 하고 싶은 데이터
    """
    sub_word = []
    sub_start = []
    sub_end = []
    sub_type = []
    
    obj_word = []
    obj_start = []
    obj_end = []
    obj_type = []
    
    for i,j in zip(data["subject_entity"], data["object_entity"]):
        s = eval(i)
        o = eval(j)
        
        s_word = s["word"]
        s_start = s["start_idx"]
        s_end = s["end_idx"]
        s_type = s["type"]
        
        o_word = o["word"]
        o_start = o["start_idx"]
        o_end = o["end_idx"]
        o_type = o["type"]
        
        sub_word.append(s_word)
        sub_start.append(s_start)
        sub_end.append(s_end)
        sub_type.append(s_type)
        
        obj_word.append(o_word)
        obj_start.append(o_start)
        obj_end.append(o_end)
        obj_type.append(o_type)

    data["sub_word"] = sub_word
    data["sub_start"] = sub_start
    data["sub_end"] = sub_end
    data["sub_type"] = sub_type
    
    data["obj_word"] = obj_word
    data["obj_start"] = obj_start
    data["obj_end"] = obj_end
    data["obj_type"] = obj_type
    

def label_to_num(data, label2num):
    """
    data의 label을 숫자로 encoding하는 함수
    """
    encoded_label = []
    for label in list(data["label"]):
        encoded_label.append(label2num[label])
    
    data["encoded_label"] = encoded_label
    
    return data
    
    
def seperate_train_val(config, train_data, val_data):
    """
    train data와 validation data를 간단하게 분리하는 함수
    """
    if config.val_data_flag == 0:
        train_data, val_data = train_test_split(train_data, test_size=0.06, random_state=6)
    elif config.val_data_flag == 1:
        train_store = []
        val_store = []
        for i in range(30):
            now_data = train_data.loc[train_data["encoded_label"] == i]
            percent = 20 / len(now_data)
            train, val = train_test_split(now_data, test_size=percent, random_state=6)
            
            train_store.append(train)
            val_store.append(val)
        
        train_data = train_store[0]
        val_data = val_store[0]
        for i in range(1, 30):
            train_data = pd.concat([train_data, train_store[i]], axis = 0)
            val_data = pd.concat([val_data, val_store[i]], axis = 0)

        train_data = train_data.sample(n=len(train_data), replace=False)
        val_data = val_data.sample(n=len(val_data), replace=False)
        
    return train_data, val_data
