from transformers import pipeline


class EmotionModel:
    def __init__(self, model):
        self.emotion_classifier = pipeline("text-classification", model=model, return_all_scores=True)

    def get_emotion(self, text):
        lst = self.emotion_classifier(text)[0]
        return max(lst, key=lambda x: x['score'])

    def get_emotions(self, text):
        return self.emotion_classifier(text)[0]

    def get_emotion_label(self, text):
        lst = self.emotion_classifier(text)[0]
        return max(lst, key=lambda x: x['score'])["label"]

    def get_emotion_list(self):
        return [dictionary["label"] for dictionary in self.emotion_classifier("Hello World")[0]]
