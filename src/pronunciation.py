import librosa
import numpy as np
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

TARGET_SR = 16000
MODEL = "masumtechnonext/wav2vec2-arabic-letter-verifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_TO_LETTER = {
    "Ain": "ع",
    "Alif": "ا",
    "Ba": "ب",
    "Daad": "ض",
    "Dal": "د",
    "Dhaa": "ظ",
    "Dhal": "ذ",
    "Faa": "ف",
    "Ghain": "غ",
    "Ha": "ه",
    "Haa": "ح",
    "Jeem": "ج",
    "Kaf": "ك",
    "Kha": "خ",
    "Laam": "ل",
    "Meem": "م",
    "Noon": "ن",
    "Qaf": "ق",
    "Raa": "ر",
    "Saad": "ص",
    "Seen": "س",
    "Sheen": "ش",
    "Ta": "ت",
    "Taa": "ط",
    "Tha": "ث",
    "Unknown": "Unknown",
    "Waw": "و",
    "Yaa": "ي",
    "Zay": "ز"
}


class PronunciationEvaluator:

    def __init__(self, threshold=75):
        self.threshold = threshold
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL).to(device)
        self.model.eval()

    def preprocess_audio(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio, _ = librosa.effects.trim(audio, top_db=25)

        if len(audio) == 0:
            raise ValueError("لم يتم اكتشاف صوت.")

        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = audio / peak

        if sample_rate != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)

        return audio

    def evaluate(self, audio_path, target_letter):
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        audio = self.preprocess_audio(audio, sr)
        inputs = self.processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(device)).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probabilities))
        predicted_label = self.model.config.id2label[pred_id]
        predicted_letter = LABEL_TO_LETTER.get(predicted_label, "Unknown")

        target_label = next((label for label, letter in LABEL_TO_LETTER.items() if letter == target_letter), None)
        target_id = next((int(idx) for idx, label in self.model.config.id2label.items() if label == target_label), None)

        score = float(probabilities[target_id]) * 100 if target_id is not None else 0
        success = predicted_letter == target_letter and score >= self.threshold

        return {
            "success": success,
            "score": round(score, 2),
            "target_letter": target_letter,
            "predicted_letter": predicted_letter,
            "message": "أحسنت! نطقت الحرف بشكل صحيح" if success else "أعد التسجيل وحاول مرة أخرى."
        }