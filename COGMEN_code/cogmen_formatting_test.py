import os
import pickle

txt_label_zip_toteval_root='/workspace/data1/emotion_competition/Code/COGMEN_code/data/KEMDy19/new_2019cogmen_format_speaker_only_feat.pkl'
# txt_label_zip_toteval_root='/workspace/data1/emotion_competition/dataset/2019cogmen_format_path.pkl'
if os.path.isfile(txt_label_zip_toteval_root):
  with open(txt_label_zip_toteval_root,'rb') as f:
    features=pickle.load(f)
print(features[8])

for i in range(len(features)):
  print(len(features[i]))
# print((features[0]))
# print((features[1]))
print((features[3][features[0][0]][30].size))
print((features[4][features[0][0]][30].size))
print(len(features[3][features[0][0]]))
print(len(features[4][features[0][0]]))


# print(features[4])
# print(len(features[4].values()))
# for script in features[0]:
#   print(len(features[1][script]))
#   print(len(features[2][script]))
#   print(len(features[3][script]))
#   print(len(features[4][script]))
#   print(len(features[6][script]),'\n')
# print(len(features[1]['Sess19_script02']))
# print(len(features[2]['Sess19_script02']))
# print(len(features[6]['Sess19_script02']))

# print((features[3]['Sess19_script02'][0].shape))
# print((features[4]['Sess19_script02'][0].shape))
