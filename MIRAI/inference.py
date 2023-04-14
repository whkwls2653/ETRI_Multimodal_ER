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
from sklearn import metrics
import argparse
from torch import nn
import numpy as np
from glob import glob
from torchsummary import summary

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_save_path', required=True, type=str)
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--clip_length', type=int, default=25)
    p.add_argument('--test_1920', type=bool, default=False)
    p.add_argument('--ckpt_path', type=str, default='')
    p.add_argument('--csv_path', type=str, default='./data/annotation.csv')
    p.add_argument('--using_model', type=str, default='./data/annotation.csv')
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
    print('preds',np.argmax(preds,axis=1))
    print('labels',labels)
    f1 = f1_score(labels, np.argmax(preds, axis=1), average='weighted')
    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    # return f1, acc
    return np.argmax(preds,axis=1),labels
def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
    text_inputs, audio_inputs, labels = batch
    pred = self.forward(text_inputs, audio_inputs)
    pred = nn.functional.softmax(pred, dim=1)
    return pred, labels
    
def main(args):
    data_config = HF_DataConfig(
        csv_path=args.csv_path
    )
    train_config = HF_TrainConfig(
        batch_size=args.batch_size,
        using_model=args.using_model

    )
    print('load from csv : ', data_config.csv_path)
    csv = pd.read_csv(data_config.csv_path)
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)

    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%args.clip_length)
    if args.test_1920 == True:
        print('use 19,20session as test')
        _, test = train_test_split(csv, test_size=978, shuffle=False)
    print(csv['emotion'][-978:])
    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)

    test_dataset = multimodal_dataset(test, data_config)
    test_loader = DataLoader(test_dataset, train_config.batch_size, num_workers=8,
                                collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                                shuffle=False, drop_last=True)
    trainer = Trainer(#gpus=1,
                    logger=False)

    model_name = ['audio', 'both', 'text']
    dict_ls = []
    """"
    root_path = sorted(glob(os.path.join(args.model_save_path,'*_multiloss/*')))
    
    
    
    for i in range(3):
        train_config = HF_TrainConfig(
            batch_size=args.batch_size,
            using_model=model_name[i]
        )
        f1, acc = predict(trainer, test_loader, train_config, root_path[i])
        dict_ls.append({
            "model" : model_name[i],
            "loss" : "cross_entropy and cosine_simliarity",
            "accuracy": acc,
            "f1_score": f1,
        })
    """
    # root_path = [sorted(glob(os.path.join(args.model_save_path,'*_ce/*')))[-1]]
    root_path=args.ckpt_path
    print(root_path)
    for i in range(1):
        train_config = HF_TrainConfig(
            batch_size=args.batch_size,
            using_model='both'
        )
        # f1, acc = predict(trainer, test_loader, train_config, root_path, loss='ce')

        preds,labels=predict(trainer, test_loader, train_config, root_path, loss='ce')
        with open('./labels_preds.txt', 'w+') as f:
            f.write(str(labels)+'\n')
            f.write(str(preds))
        print(metrics.classification_report(labels, preds, digits=4))
        # dict_ls.append({
        #     "model" : args.ckpt_path.split('/')[-2],
        #     "loss" : "cross_entropy",
        #     "accuracy": acc,
        #     "f1_score": f1,
        # })
    """"   
    root_path = sorted(glob(os.path.join(args.model_save_path,'both_multiloss_MMI/*')))
    
    dict_ls = []
    f1, acc = predict_MMI(trainer, test_loader, train_config, root_path[0])
    dict_ls.append({
        "model" : "both_MMI",
        "loss" : "cross_entropy and cosine_simliarity",
        "accuracy": acc,
        "f1_score": f1
    })
    
    root_path = sorted(glob(os.path.join(args.model_save_path,'both_ce_MMI/*')))
    f1, acc = predict_MMI(trainer, test_loader, train_config, root_path[0], loss='ce')
    dict_ls.append({
        "model" : "both_MMI",
        "loss" : "cross_entropy",
        "accuracy": acc,
        "f1_score": f1,
    })
    """
    # pd.DataFrame(dict_ls).to_csv('./result.csv', index=False)
# %%
if __name__ == '__main__':
    args = define_argparser()
    main(args)