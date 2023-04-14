from sentence_transformers import SentenceTransformer


sbert_model = SentenceTransformer("/workspace/data1/emotion_competition/Code/COGMEN_code/cogmen/sbert")


class Sample:
    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
        self.sbert_sentence_embeddings = sbert_model.encode(sentence)
