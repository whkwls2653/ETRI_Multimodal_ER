ETRI_Multimodal_ER
=================
* ETRI 멀티모달 감정인식 대회
* Data: KEMDy19
* Modality : Audio, Text 
* Title : 'Contextualized GNN구조를 활용한 한국어 대화에서의 멀티모달 감정인식'

<img src="/structure.png" width="600px" height="400px" title="structures" alt="structures"></img><br/>

본 방법론의 구조는 크게 다음과 같이 구성된다.  
### 1. Feature extractor trainning
### 2. Feature formatting to CGNN Format
### 3. CGNN 학습  
빠른 추론을 위해 1,2번 과정을 논문에 적힌대로 수행하여  
피클링후 분할압축 해놓아 3번부터도 진행이 가능하니  
빠른 추론을 원하는 사람은 환경구축 이후에 바로 3번 if부터 진행하길 바람.
# 0. 환경 구축
## requirements
### Codes
<pre>
<code>
git clone https://github.com/whkwls2653/ETRI_Multimodal_ER /your/path
</code>
</pre>
### dataset
* Download 'KEMDy19' and put it /dataset/
### libraries
* torch
* transformers
* pytorch-lightning
* PyTorch Geometric
* Comet 
* pandas
* librosa
* sklearn
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>
### pretrained models
model 안에서 자동으로 인터넷에서 받아짐.
* sbert : 'paraphrase-distilroberta-base-v1'
* klue : 'klue/roberta-base'
* wav2vec2 : 'w11wo/wav2vec2-xls-r-300m-korean'

# 1. Feature extractor trainning 
run_command.sh 명령어들 참조.  
--exp_name : 실험 제목, 결과물 폴더명.  
--using_model : feature extractor 학습시 사용하는 모달리티, 본 논문에서는 both 고정  
--test_1920 : CGNN 테스트셋을 feature extractor 학습의 데이터셋으로 사용하지 않기위해 test set을 19,20 session으로 고정  
<pre>
<code>
python trainer_hf.py --exp_name both_test_1920_speakeronly --using_model both --batch_size 2 --accumulate_grad 8 --test_1920=True --csv_path="./data/annotation_speaker_only.csv"
</code>
</pre>

# 2. Feature formatting to CGNN Format
주의)2번 단계는 MIRAI/config.py의 feature_extract: bool =True로 바꾸어 진행  
--model_save_path : ./models_zoo/checkpoint/에서 feature extractor로 사용하고자 하는 ckpt 설정  
<pre>
<code>
cd COGMEN_Code
python cogmen_formatting.py
cd ../MIRAI
CUDA_VISIBLE_DEVICES=0 python feature_extractor_cogmen_format.py --features_format_path='../COGMEN_code/data/KEMDy19/new_2019cogmen_format_speaker_only.pkl'  --model_save_path='/your/ckpt/path' --csv_path='./data/annotation_speaker_only.csv'
</code>
</pre>

# 3. CGNN 학습 및 추론
## if)3번부터 시작하기 -> 피클파일 unzip
<pre>
<code>
apt-get install zip unzip
cd COGMEN_code/data/KEMDy19
zip -s 0 zip.zip --out unziptest.zip
unzip unziptest.zip
cd ../../model_checkpoints
zip -s 0 zip.zip --out unziptest.zip
unzip unziptest.zip
</code>
</pre>
##  preprocess
preprocess, Train, Evaluation실행 관련해선 COGMEN_code/run_eval.sh 참조  
cogmen formatting.py로 나온 Csession, test, train 정보를 바탕으로 pkl file 형성  
* --res_dir : preprocessed pkl file 생성 위치
* --feat_dir : feature extract하여 cogmen format으로 맞춰 피클링한 파일 위치  
* --dataset : 사용할 데이터셋 설정, 본 논문에선 'KEMDy19' 고정   
* 현재 디렉토리 확인해서 github root에 있도록 해야함.
<pre>
<code>
cd COGMEN_code
python preprocess.py --res_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --feat_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat.pkl' --dataset='KEMDy19'
</code>
</pre>
## Train
preprocessed된 데이터로 CGNN학습 진행.
* --tag : 실험 명을 입력. 학습된 ckpt가 나오는 디렉토리명 결정함.  
* --dataset : 사용할 데이터셋 설정, 본 논문에선 'KEMDy19' 고정  
* --modalities : 실험에 사용하는 모달리티 결정. at, t, a중 설정  
* --from_begin : 처음부터 학습  
* --epochs : 학습에폭 수  
<pre>
<code>
python train.py --tag='tmp' --dataset='KEMDy19' --modalities='at' --preprocessed_feature='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --from_begin --epochs=100
</code>
</pre>
## Evaluation
학습중 저장된 validation set의 F1이 가장 높은 모델로 테스트셋 evaluation 진행
* --dataset : 사용할 데이터셋 설정, 본 논문에선 'KEMDy19' 고정
* --modalities : 실험에 사용하는 모달리티 결정. at, t, a중 설정
* --data_dir : evaluation에 사용되는 preprocessed된 pkl파일 설정. 해당 pkl에 test 발화 정보가 매핑되어 있어 스스로 test셋 꾸려 사용.
<pre>
<code>
python eval.py --dataset="KEMDy19" --modalities="at" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_at_MIRAI_pretrained_speaker_only.pt'
python eval.py --dataset="KEMDy19" --modalities="a" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_a_MIRAI_pretrained_speaker_only.pt'
python eval.py --dataset="KEMDy19" --modalities="t" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_t_MIRAI_pretrained_speaker_only.pt'
</code>
</pre>

<img src="/results.png" width="600px" height="400px" title="structures" alt="structures"></img><br/>

# citation
* [1] K. J. Noh and H. Jeong, “KEMDy19,” https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR 
* [2] Joshi, Abhinav, et al. "COGMEN: COntextualized GNN based multimodal emotion recognitioN." arXiv preprint arXiv:2205.02455 (2022).
# acknowledgements
* The structure of our code is inspired by <https://github.com/Exploration-Lab/COGMEN>
* and <https://github.com/Mirai-Gadget-Lab/Multimodal_Emotion_Recognition>
