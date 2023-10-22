import datasets
import os
import random
import re
from abc import ABC, abstractmethod
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from eda import eda

import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw


class Augmenter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, example, num_aug):
        pass


class NlpAugAugmenter(Augmenter):
    def __init__(self):
        super().__init__()
        self.aug = None

    def __call__(self, example, num_aug):
        return [self.aug.augment(example) for i in range(num_aug)]


class ContextualAugmenter(NlpAugAugmenter):
    def __init__(self, aug_p=0.3, aug_min=1):
        super().__init__()
        self.aug = naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased',
            action="substitute",
            aug_p=aug_p,
            aug_min=aug_min, aug_max=None)



class RandomAugmenter(NlpAugAugmenter):
    def __init__(self, action='delete', aug_p=0.2, aug_min=1, target_words=None):
        super().__init__()
        self.aug = naw.RandomWordAug(action=action, aug_p=aug_p, aug_max=None,
                                     aug_min=aug_min, target_words=target_words)


class TFIDFReplacementAugmenter(NlpAugAugmenter):
    def __init__(self, dataset_name, data,
                 aug_p=0.2, aug_min=1):
        super().__init__()

        self.model_path = os.path.join('cache', dataset_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            # train tf_idf model
            train_tokens_data = [self._tokenize(t) for t in data]
            tfidf_model = nmw.TfIdf()
            tfidf_model.train(train_tokens_data)
            tfidf_model.save(self.model_path)

        self.aug = naw.TfIdfAug(self.model_path,
                                aug_p=aug_p, aug_min=aug_min, aug_max=None,
                                tokenizer=self._tokenize)

    def _tokenize(self, text, token_pattern=r'(?u)\b\w\w+\b'):
        token_pattern = re.compile(token_pattern)
        return token_pattern.findall(text)


class EDAAugmenter(Augmenter):
    def __init__(self, aug_p=0.2):
        super().__init__()
        self.aug_p = aug_p

    def __call__(self, example, num_aug):
        return eda(example, alpha=self.aug_p, num_aug=num_aug)


class CropAugmenter(Augmenter):
    def __init__(self, aug_p=0.2):
        super().__init__()
        self.l_p = 1 - aug_p

    def crop(self, text):
        tokens = word_tokenize(text)
        crop_l = int(self.l_p * len(tokens))
        start_pos = random.randint(0, len(tokens) - crop_l)
        cropped_text = ' '.join(tokens[start_pos:start_pos+crop_l])
        return cropped_text

    def __call__(self, example, num_aug):
        return [self.crop(example) for _ in range(num_aug)]


class Compositional(Augmenter):
    def __init__(self, sequential_augmenters):
        super().__init__()
        self.augmenters = sequential_augmenters

    def __call__(self, example, num_aug):
        ret = []
        for i in range(num_aug):
            tmp = example
            for aug in self.augmenters:
                tmp = aug(tmp)
            ret.append(tmp)
        return ret


if __name__ == '__main__':
    sample_text = 'will find little of interest in this film, which is often preachy and poorly acted '

    first_augmenter = EDAAugmenter()
    # second_augmenter = CropAugmenter()
    # augmenter = Compositional([first_augmenter, second_augmenter])
    for i in range(10):
        print(first_augmenter({'text': [sample_text]}))
        print()
