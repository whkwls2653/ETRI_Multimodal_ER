import pandas as pd
import os
from dataset_hf import multimodal_dataset, multimodal_collator
from torch.utils.data import DataLoader
from config import *
from models.pl_model_hf import *
from transformers import AutoTokenizer, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score, accuracy_score
import argparse
from torch import nn
from glob import glob
import pickle as pkl
import torch
import numpy as np
def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_save_path', type=str,default='./models_zoo/checkpoint/both_test_1920_speakeronly/epoch=00-val_loss=0.74870.ckpt')
    p.add_argument('--batch_size', default=6, type=int)
    p.add_argument('--clip_length', type=int, default=25)
    p.add_argument('--feature_extract', type=bool, default=True)
    p.add_argument('--features_format_path', type=str, default='./')
    p.add_argument('--csv_path', type=str, default='./')
    config = p.parse_args()

    return config

def predict_MMI(trainer, loader, train_config, ckp_path, loss=None):
    if loss == 'ce':
        model = PL_model_MMER(train_config).load_from_checkpoint(ckp_path)
    else:
        model = PL_model_MMER_multiloss(train_config).load_from_checkpoint(ckp_path)
    model.predict_step = predict_step.__get__(model)
    predictions = trainer.predict(model, loader)
    preds = [i[0] for i in predictions]
    labels = [i[1]['emotion'] for i in predictions]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    f1 = f1_score(labels, np.argmax(preds, axis=1), average='weighted')
    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    return f1, acc

def predict(trainer, loader, train_config, ckp_path, loss=None):
    if loss == 'ce':
        model = PL_model_ce(train_config).load_from_checkpoint(ckp_path)
    else:
        model = PL_model(train_config).load_from_checkpoint(ckp_path)
    model.predict_step = predict_step.__get__(model)
    predictions = trainer.predict(model, loader)
    preds = [i[0] for i in predictions]
    labels = [i[1]['emotion'] for i in predictions]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    feats=preds
    print('feats',feats.shape,feats)
    print('labels',labels.shape,labels)
    return feats

def feautre_extractor(trainer, loader, train_config, ckp_path, loss=None):
    if loss == 'ce':
        model = PL_model_ce(train_config).load_from_checkpoint(ckp_path)
    else:
        model = PL_model(train_config).load_from_checkpoint(ckp_path)
    model.predict_step = predict_step.__get__(model)
    print('before prediction')
    predictions = trainer.predict(model, loader)
    loader_feats = [i[0] for i in predictions]
    labels = [i[1]['emotion'] for i in predictions]
    # loader_feats = torch.cat(loader_feats)
    # labels = torch.cat(labels)
    return loader_feats,labels
def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
    text_inputs, audio_inputs, labels = batch
    pred = self.forward(text_inputs, audio_inputs)
    pred = nn.functional.softmax(pred, dim=1)
    return pred, labels
    
def main(args):
    data_config = HF_DataConfig(csv_path=args.csv_path)
    train_config = HF_TrainConfig(
        batch_size=args.batch_size,
        feature_extract=True
    )

    features_format_path = args.features_format_path
    # features_res_path=args.features_res_path
    if os.path.isfile(features_format_path):
        with open(features_format_path, 'rb') as f:
            features_format = pkl.load(f)
    script_list = features_format[0]
    mf_dict = features_format[1]
    labels_dict = features_format[2]
    string_dict = features_format[6]
    f_name_dict = features_format[9]

    ### find numbers to make structure
    numbers = []
    numbers_accum = []
    for script in script_list:
        # index starts from 1
        x = 0
        for i in range(len(f_name_dict[script])):
            x += 1
        numbers.append(x)
    total = 0
    for i in numbers:
        total += i
        numbers_accum.append(total)


    #####
    ##feature extract with csv
    csv = pd.read_csv(data_config.csv_path)

    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)
    print('csv len:',len(csv))
    ### use test set as whole dataset to extract features

    test_dataset = multimodal_dataset(csv, data_config)
    print('test_dataset :', len(test_dataset), test_dataset)
    test_loader = DataLoader(test_dataset, train_config.batch_size, num_workers=8,
                                collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                                shuffle=False, drop_last=False)
    print('test_loader',len(test_loader),test_loader)
    trainer = Trainer(#gpus=1,
                    logger=False)



    # root_path = [sorted(glob(os.path.join(args.model_save_path,'*_ce/*')))[-1]]

    print('load model from ckpt :',args.model_save_path)


    feats,labels=feautre_extractor(trainer, test_loader, train_config, args.model_save_path, loss='ce')
    feat_label=[torch.cat(feats),torch.cat(labels)]
    print(feat_label)
    print('over')
    # print('feats',feats,len(feats))
    # with open(features_res_path, 'wb') as f:
    #     pkl.dump(feat_label, f)
    #     # print(predict(trainer, test_loader, train_config, root_path[i], loss='ce'))
    #
    #
    # if os.path.isfile(features_res_path):
    #     with open(features_res_path, 'rb') as f:
    #         features = pkl.load(f)
    # # print(features[0])
    # # print(features[1])
    # # features[0] = torch.cat(torch.tensor(features[0]))
    # # features[1] = torch.cat(torch.tensor(features[1]))
    # print(features[0])
    # print(features[1])
    features=feat_label
    i = 0
    j = 0
    print(numbers_accum)
    for i in range(len(features[1])):
        if i in numbers_accum:
            j += 1
            # print(len(features[3][script]))
            # print(len(features[4][script]))

        script = script_list[j]
        # print(j)
        # print('script:', script)
        # print(txt_feats)
        features_format[3][script].append(np.array(features[0][i][:768]))
        # print(features[3])
        features_format[4][script].append(np.array(features[0][i][768:]))
    # print('txt_feats',len(txt_feats))
    # print('wav_feats',len(wav_feats))

    # for txt_feat, wav_feat in zip(txt_feats, wav_feats):
    #     # print('txt_feat shape:',txt_feat.shape)
    #     # print('wav_feat shape:',wav_feat.shape)
    #


# print(features[3])
#     tmp='/workspace/data1/emotion_competition/Code/COGMEN_code/data/KEMDy19/tmp_feat.pkl'
    with open(features_format_path, 'wb') as f:
        pkl.dump(features_format, f)
        # dict_ls.append({
        #     "model" : model_name[i],
        #     "loss" : "cross_entropy",
        #     "accuracy": acc,
        #     "f1_score": f1,
        # })
#     """"
#     root_path = sorted(glob(os.path.join(args.model_save_path,'both_multiloss_MMI/*')))
#
#     dict_ls = []
#     f1, acc = predict_MMI(trainer, test_loader, train_config, root_path[0])
#     dict_ls.append({
#         "model" : "both_MMI",
#         "loss" : "cross_entropy and cosine_simliarity",
#         "accuracy": acc,
#         "f1_score": f1
#     })
#
#     root_path = sorted(glob(os.path.join(args.model_save_path,'both_ce_MMI/*')))
#     f1, acc = predict_MMI(trainer, test_loader, train_config, root_path[0], loss='ce')
#     dict_ls.append({
#         "model" : "both_MMI",
#         "loss" : "cross_entropy",
#         "accuracy": acc,
#         "f1_score": f1,
#     })
#     """
#     # pd.DataFrame(dict_ls).to_csv('./result.csv', index=False)
# %%
if __name__ == '__main__':
    args = define_argparser()

    main(args)