from dataclasses import dataclass

@dataclass
class HF_DataConfig():
    """
    Data Settings
    """
    root_path: str = "./data/txt_wav"
    csv_path: str = "./data/annotation.csv"
    normalized: bool = True
    remove_non_text: bool = True
    # return_text: bool = False
    
@dataclass
class HF_TrainConfig():
    lr: float = 5e-5
    label_name: str = 'emotion'
    checkpoint_path: str = './models_zoo/checkpoint/'
    log_dir: str = './models_zoo/tensorboard/'
    using_model: str = 'both'
    batch_size: int = 4
    text_encoder: str = "/workspace/data1/emotion_competition/Code/MIRAI/pretrained/klue_roberta_tot"
    audio_processor: str = "/workspace/data1/emotion_competition/Code/MIRAI/pretrained/wav2vec2_korean_tot"
    feature_extract: bool = False