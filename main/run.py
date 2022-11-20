import subprocess

class Config:
    def __init__(self):
        ## 1. 모델 학습시킬 때 가장 먼저 바꿔야 하는 것
        self.model_name: str = "jhgan/ko-sbert-multitask"
        self.save_path: str = "../../saved_model/sbert_model.pt" # 최종 모델을 어디에 저장할지
        self.result_path: str = "../../dataset/submission.csv" # 마지막 csv 파일을 어디에 저장할지
        self.wandb_name: str = "jhgan/ko-sbert-multitask" # wandb 내에서 작업 이름 설정 (중요)
        
        ## 2. 데이터 위치
        self.train_data_path: str = "../../dataset/train/train2/train_80.csv"
        self.val_data_path: str = "../../dataset/valid/valid2/valid_20.csv"
        self.test_data_path: str = "../../dataset/test/test_data.csv"

        ## 3. 학습 설정
        self.epoch: int = 5
        '''
        0 : base
        1 : binary classification
        '''
        self.model_type: int = 0
        self.input_type: int = 0

        ## 4. 모델 하이퍼파라미터
        self.num_hidden_layer: int = 0
        self.mx_token_size: int = 256
        self.batch_size: int = 16
        self.lr: float = 5e-5

        ## 5. 한번 바꾸면 바꿀일 없는 설정
        self.wandb_project: str = "KLUE_RE"
        self.wandb_entity: str = "happy06"
        self.wandb_group: str = "SBERT"

def cmd_parser(dic):
    cmd = "python3 main.py "
    for key in dic:
        param = "--{}={} ".format(key, dic[key])
        cmd += param
    return cmd

if __name__ == "__main__":
    config = Config()
    dic = config.__dict__

    cmd = cmd_parser(dic)
    
    subprocess.call(cmd, shell=True)