import numpy as np
import pooch
from fastai.vision.all import load_learner
from PIL import Image


class DogBreedsDetection:
    # TODO: provide docstring
    
    def __init__(self):
        self.url = (
            "https://github.com/ironcladgeek/ml2prod/releases/"
            "download/v0.0.0/dog_breeds__swin_tiny_v1.pkl"
        )
        self.md5 = "4e8a2b900244823e1dac2a5c2f408df5"

    def load_model(self, cpu=True):
        self.model_path = pooch.retrieve(
            url=self.url, known_hash=f"md5:{self.md5}", progressbar=True
        )
        return load_learner(fname=self.model_path, cpu=cpu)

    @staticmethod
    def top_k(vocab, probs, k=5):
        _vocab = np.array(vocab)
        _probs = np.array(probs)
        sort_idx = np.argsort(_probs)
        sorted_vocab = _vocab[sort_idx][::-1]
        sorted_probs = _probs[sort_idx][::-1]
        return dict(zip(sorted_vocab[:k], sorted_probs[:k]))

    def predict(self, img_fp, k=5):
        img = np.array(Image.open(img_fp).convert("RGB"))
        model = self.load_model()
        _, _, probs = model.predict(img)
        top_k_preds = self.top_k(vocab=model.dls.vocab, probs=probs, k=k)
        return top_k_preds
