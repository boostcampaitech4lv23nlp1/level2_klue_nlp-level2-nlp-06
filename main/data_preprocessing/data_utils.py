import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Get weight from label distribution for 'weighted CrossEntropy'
def get_weights_prob(label2num: dict, df: pd.DataFrame, weighted=0):
    """
    CrossEntropy loss의 label 별 loss weight 계산하는 함수

    Args:
        label2num (dict): label to num dictionary
        df (pd.DataFrame): dataset
        weighted (int, optional): loss type

    Returns:
        torch.Tensor: label 별 loss weight
    """    
    if weighted == 1:
        weights = []
        
        for i, label in enumerate(label2num.keys()):
            cnt = len(df[df["label"]==label])
            weights.append(cnt**2)

        weights = [(1 - (weight/sum(weights))) for weight in weights]
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
