import subprocess


class Config:
    def __init__(self):
        ## 1. 모델 학습시킬 때 가장 먼저 바꿔야 하는 것
        self.model_name: str = "klue/roberta-large"
        self.save_path: str = "../../saved_model/tt.pt" # 최종 모델을 어디에 저장할지
        self.result_path: str = "../../dataset/submission.csv" # 마지막 csv 파일을 어디에 저장할지
        self.wandb_name: str = "'roberta-large: total'" # wandb 내에서 작업 이름 설정 (중요)
        self.wandb_group: str = "is_related?" # wandb group.
        self.wandb_note: str = "'roberta-large로 그냥 모두 train한 경우'" # wandb note
        
        ## 2. 데이터 위치 (csv 파일)
        self.train_data_path: str = "../../dataset/train/train8/total_train.csv"
        self.val_data_path: str = "../../dataset/valid/valid8/total_valid.csv"
        self.test_data_path: str = "../../dataset/test/test_data.csv"

        ## 3. 학습 설정
        '''
        train_type = {0: "base-model", 1: "rescent"}
        model_type = {0: "base-model", 1: "Masked_QA"}
        input_type = {0: "base-input", 1: "typed_punct_entity", 2: "Masked_QA", 3:"typed_punct_entity_front", 4: "entity_mask", 5: "entity_marker", 6:"typed_entity_marker"} 
        
        '''
        self.train_type: int = 0
        self.model_type: int = 1
        self.input_type: int = 1
        self.pooling = "MEAN" # 어떤 pooler output을 사용할 것인지 : ["MEAN", "CLS"]
        self.rnn_type: str = 'lstm' #['lstm', 'gru', None(사용 안함)] - rnn_type 사용 시 pooling은 적용되지 않음. 참고.
        self.epoch: int = 3
        self.checkpoint_dir: str = "./results/rescent/roberta-large/total" # Trainer의 학습 checkpoint 저장 경로.
        self.label_dict_dir: str = "./results/rescent/roberta-large/total/label2num.pickle" # RESCENT : label2num dictionary save path.
        self.warmup_step: int = 500 # learning rate warmup step.
        self.eval_step: int = 500 # 모델 평가/저장 step 수.

        ## 4. 모델 하이퍼파라미터
        self.num_hidden_layer: int = 0 # BERT 뒤에 linear layer를 몇 개 쌓을지.
        self.mx_token_size: int = 256 # 문장 최대 길이
        self.batch_size: int = 32
        self.lr: float = 3e-5
        self.weighted_loss: bool = True # True로 설정할 경우 label distribution에 따라 weighted loss를 수행.

        ## 5. 한번 바꾸면 바꿀일 없는 설정
        self.wandb_project: str = "KLUE_RE"
        self.wandb_entity: str = "happy06"
        '''
        input type 에 대한 자세한 설명
        input type = 0  base_input                          :   박수현 [SEP] 시청 [SEP] 박수현은 오늘 시청에 들렀다.
        input type = 1  typed_entity_marker_punct_kr        :   @ + 사람 + 박수현 @ 은 오늘 # ^ 장소 ^ 시청 # 에 들렀다
        input type = 2  Masked_QA                           :   박수현은 오늘 시청에 들렀다 [SEP] 박수현와 시청의 관계는 [MASK]
        input type = 3  typed_entity_marker_punct_kr_front  :   @ + PER + 박수현 @[SEP]# ^ LOC ^ 시청 #[SEP] @ + PER + 박수현 @ 은 오늘 # ^ LOC ^ 시청 # 에 들렀다
        input type = 4  entity_mask                         :   [SUBJ-PER] 은 오늘 [OBJ-LOC] 에 들렀다
        input type = 5  entity_marker                       :   [E1] 박수현 [/E1]은 오늘 [E2] 시청 [/E2] 에 들렀다
        input type = 6  typed_entity_marker                 :   [S:PER] 박수현 [/S:PER] 은 오늘 [O:LOC] 시청 [/O:LOC] 에 들렀다.
        '''

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
