#1.preprocess
#python preprocess.py --res_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --feat_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat.pkl' --dataset='KEMDy19'

#2.Train
#python train.py --tag='tmp' --dataset='KEMDy19' --modalities='at' --preprocessed_feature='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --from_begin --epochs=100
#python train.py --tag='MIRAI_pretrained' --dataset='KEMDy19' --modalities='a' --preprocessed_feature='./data/KEMDy19/data_KEMDy19_MIRAI_pretrained.pkl' --from_begin --epochs=55
#python train.py --tag='MIRAI_pretrained' --dataset='KEMDy19' --modalities='t' --preprocessed_feature='./data/KEMDy19/data_KEMDy19_MIRAI_pretrained.pkl' --from_begin --epochs=55


#3.Evaluate

python eval.py --dataset="KEMDy19" --modalities="at" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_at_MIRAI_pretrained_speaker_only.pt'
python eval.py --dataset="KEMDy19" --modalities="a" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_a_MIRAI_pretrained_speaker_only.pt'
python eval.py --dataset="KEMDy19" --modalities="t" --data_dir='./data/KEMDy19/new_2019cogmen_format_speaker_only_feat_preprocessed.pkl' --pt_dir='./model_checkpoints/KEMDy19_best_dev_f1_model_t_MIRAI_pretrained_speaker_only.pt'






