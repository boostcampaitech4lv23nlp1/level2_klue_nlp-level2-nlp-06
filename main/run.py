import subprocess


class Config:
    def __init__(self):
        ## 1. 모델 학습시킬 때 가장 먼저 바꿔야 하는 것
        self.model_name: str = "klue/bert-base"
        self.save_path: str = "../../saved_model/PER-POH.pt" # 최종 모델을 어디에 저장할지
        self.result_path: str = "../../dataset/submission.csv" # 마지막 csv 파일을 어디에 저장할지
        self.wandb_name: str = "PER-POH" # wandb 내에서 작업 이름 설정 (중요)
        self.wandb_group: str = "Rescent" # wandb group.
        self.wandb_note: str = "'PER-POH group'" # wandb note
        
        ## 2. 데이터 위치 (csv 파일)
        self.train_data_path: str = "../../dataset/train/train6/PER-POH_train.csv"
        self.val_data_path: str = "../../dataset/valid/valid6/PER-POH_valid.csv"
        self.test_data_path: str = "../../dataset/test/test_data.csv"

        ## 3. 학습 설정
        '''
        train_type = {0: "base-model", 1: "rescent"}
        model_type = {0: "base-model", 1: "Masked_QA"}
        input_type = {0: "base-input", 1: "typed_punct_entity", 2: "Masked_QA"}
        '''
        self.train_type: int = 1
        self.model_type: int = 0
        self.input_type: int = 0
        self.epoch: int = 5
        self.checkpoint_dir: str = "./results/rescent/PER-POH" # Trainer의 학습 checkpoint 저장 경로.
        self.label_dict_dir: str = "./results/rescent/PER-POH/label2num.pickle" # RESCENT : label2num dictionary save path.
        self.warmup_step: int = 100 # learning rate warmup step.
        self.eval_step: int = 100 # 모델 평가/저장 step 수.

        ## 4. 모델 하이퍼파라미터
        self.num_hidden_layer: int = 0 # BERT 뒤에 linear layer를 몇 개 쌓을지.
        self.mx_token_size: int = 256 # 문장 최대 길이
        self.batch_size: int = 32
        self.lr: float = 5e-5

        ## 5. 한번 바꾸면 바꿀일 없는 설정
        self.wandb_project: str = "KLUE_RE"
        self.wandb_entity: str = "happy06"
        

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
