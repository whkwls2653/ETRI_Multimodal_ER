

##preprocess
#python preprocess.py --root_path='../../dataset/KEMDy19' --save_path='./data/'
###train
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name text_test_1920  --using_model text --batch_size 64 --accumulate_grad 1 --test_1920=True
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name audio_test_1920  --using_model audio --batch_size 2 --accumulate_grad 8 --test_1920=True
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name text_test_1920_speakeronly  --using_model text --batch_size 64 --accumulate_grad 1 --test_1920=True --csv_path="./data/annotation_speaker_only.csv"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name audio_test_1920_speakeronly  --using_model audio --batch_size 2 --accumulate_grad 8 --test_1920=True --csv_path="./data/annotation_speaker_only.csv"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name both_test_1920_speakeronly  --using_model both --batch_size 2 --accumulate_grad 8 --test_1920=True --csv_path="./data/annotation_speaker_only.csv"

#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name audio_ce --using_model audio --batch_size 2 --accumulate_grad 8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name test_1920  --using_model both --batch_size 2 --accumulate_grad 8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name text_ce  --using_model text --batch_size 64 --accumulate_grad 1
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name audio_multiloss --using_model audio --batch_size 2 --accumulate_grad 8 --loss cs_and_ce
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name both_multiloss  --using_model both --batch_size 2 --accumulate_grad 8 --loss cs_and_ce
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf.py --exp_name text_multiloss  --using_model text --batch_size 64 --accumulate_grad 1 --loss cs_and_ce
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf_MMER.py --exp_name both_ce_MMI --batch_size 2 --accumulate_grad 8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_hf_MMER.py --exp_name both_multiloss_MMI --batch_size 2 --accumulate_grad 8 --loss cs_and_ce


###inference
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/text_test_1920_speakeronly/epoch=02-val_loss=0.87181.ckpt --test_1920 True --csv_path ./data/annotation_speaker_only.csv
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/both_test_1920_speakeronly/epoch=00-val_loss=0.74870.ckpt --test_1920 True --csv_path ./data/annotation_speaker_only.csv
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/audio_test_1920_speakeronly/epoch=05-val_loss=0.79508.ckpt --test_1920 True --csv_path ./data/annotation_speaker_only.csv
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/both_test_1920/epoch=01-val_loss=0.87100.ckpt --test_1920 True
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/audio_test_1920/epoch=05-val_loss=0.90756.ckpt --test_1920 True --using_model='audio'
#CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/text_test_1920/epoch=01-val_loss=1.04645.ckpt --test_1920 True --using_model='text'
CUDA_VISIBLE_DEVICES=0 python inference.py --model_save_path ./models_zoo/checkpoint/ --ckpt_path ./models_zoo/checkpoint/both_test_1920_speakeronly/epoch=00-val_loss=0.74870.ckpt --test_1920 True --using_model='both' --csv_path ./data/annotation_speaker_only.csv
###feature extracor
#CUDA_VISIBLE_DEVICES=0 python feature_extractor_cogmen_format.py --features_format_path='../COGMEN_code/data/KEMDy19/new_2019cogmen_format_speaker_only_test.pkl'  --model_save_path='./models_zoo/checkpoint/both_test_1920_speakeronly/epoch=00-val_loss=0.74870.ckpt' --csv_path='./data/annotation_speaker_only.csv'
