import csv
import pandas as pd
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

## formating features dim 9 list
features=[]
sess_ids=[]
sess_speakers=dict()
sess_labels=dict()
sess_text=dict()
sess_audio=dict()
sess_visual=dict()
sess_sentence=dict()
sess_dir=dict()
sess_train=[]
sess_test=[]
## roots
annotation_root= '../MIRAI/data/annotation_speaker_only.csv'
txt_wav_root='../MIRAI/data/txt_wav'
txt_label_zip_toteval_root='./data/KEMDy19/new_2019cogmen_format_speaker_only.pkl'
##parameters
clip_length=25
emotion2int = {
  'neutral': 0,
  'angry': 1,
  'happy': 2,
  'surprise': 3,
  'sad': 4,
  'fear': 5,
  'disgust': 6
}

csv = pd.read_csv(annotation_root)
print(len(csv))
# csv['wav_length'] = csv['wav_end'] - csv['wav_start']
# csv = csv.query("wav_length <= %d" % clip_length)

###10135
print(csv['segment_id'][0])


#
# wav_path = os.path.join(root_path, csv['segment_id'].iloc[idx] + '.wav')
# txt_path = os.path.join(root_path, csv['segment_id'].iloc[idx] + '.txt')
#
# wav = self._load_wav(wav_path)
# txt = self._load_txt(txt_path)
emotion_list=[]
csv_emotion_list = csv['emotion']
print(len(csv_emotion_list))
# print(emotion_list[emotion2int])
for emotion in csv_emotion_list:
  emotion_list.append(emotion2int[emotion])
segment_id_list=csv['segment_id']
print(len(emotion_list))
print(len(segment_id_list))
# print(emotion_list)
# valence = csv['valence']
# arousal = csv['arousal']
for i,segment_id in enumerate(segment_id_list):
  all_txts=[]
  f_name=segment_id
  gender = f_name.split('_')[-1][0]
  sess_num = f_name.split('_')[0][-2:]
  sc_pro_num = f_name.split('_')[0] + '_' + f_name.split('_')[1]
  wav_loc=os.path.join(txt_wav_root,(segment_id+'.wav'))
  txt_loc=os.path.join(txt_wav_root,(segment_id+'.txt'))
  emotion=emotion_list[i]
  ### get txt contents with deletion of strange character
  with open(txt_loc) as f:
    lines = f.readlines()[0]
    if '\n' in lines or'/'in lines :
      # print('lines',lines)
      new_words=lines.replace('\n','')
      # print('new_words',new_words)
      new_words=new_words.replace('o/','')
      new_words=new_words.replace('I/','')
      new_words=new_words.replace('b/','')
      new_words=new_words.replace('u/','')
      new_words=new_words.replace('l/','')
      new_words=new_words.replace('n/','')
      new_words=new_words.replace('N/','')
      new_words=new_words.replace('s/','')
      new_words=new_words.replace('c/','')
      new_words=new_words.replace('(','')
      new_words=new_words.replace(')','')
      new_words=new_words.replace('+',' ')
      lines=new_words
  ##make dict
  if sc_pro_num not in sess_ids:
    sess_ids.append(sc_pro_num)
    sess_speakers[sc_pro_num]=[]
    sess_labels[sc_pro_num]=[]
    sess_text[sc_pro_num]=[]
    sess_audio[sc_pro_num]=[]
    sess_visual[sc_pro_num]=[]
    sess_sentence[sc_pro_num]=[]
    sess_dir[sc_pro_num]=[]
    if int(sess_num) ==19 or int(sess_num)==20:
      sess_test.append(sc_pro_num)
    else:
      sess_train.append(sc_pro_num)

  sess_labels[sc_pro_num].append(emotion)
  sess_speakers[sc_pro_num].append(gender)
  sess_dir[sc_pro_num].append(f_name)
# print('labels',labels,type(labels))
# print('stcs',lines,type(lines))
# sess_labels[sc_pro_num]=file_labels
  sess_sentence[sc_pro_num].append(lines)

features.append(sess_ids)
features.append(sess_speakers)
features.append(sess_labels)
features.append(sess_text)
features.append(sess_audio)
features.append(sess_visual)
features.append(sess_sentence)
features.append(sess_train)
features.append(sess_test)
features.append(sess_dir)
# # print(features)
with open(txt_label_zip_toteval_root,'wb') as f:
  pickle.dump(features,f)
# # print("data_list length :",len(data_list))
# # print("sample : ",data_list[0])
# print('ambig_annotation:',ambig_annotation)
# print('no_file',no_file)
# # print(len(all_txts))
