import os
import csv
import math
import pandas as pd
import ray
import multiprocessing
import modin.pandas as mpd
from dataclasses import dataclass, field

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


@dataclass(eq=False)
class BleuScoreMetrics:
    local_bucket_augmented_sms_path: str = field(
        default=None, metadata={"help": "local augment data path througe bleuscore"}
    )
    tokenizer: PreTrainedTokenizerBase = field(
        default=None, metadata={"help": "Huggingface PreTrainedTokenizerBase object"}
    )
    
    def __post_init__(self):
        self.augment_path = os.path.dirname(self.local_bucket_augmented_sms_path)
        self.original_sms_path = os.path.join(self.augment_path, 'original_sms.csv')
        self.augment_sms_path = os.path.join(self.augment_path, 'augment_sms.csv')
        self.smooth = SmoothingFunction().method2

    def parse_to_dataframe(self):
        original_sms_df = pd.read_csv(self.original_sms_path, names=['original_content', 'labels'])
        augment_sms_df = pd.read_csv(self.augment_sms_path, names=['augment_content', 'labels'])
        print('original_sms_df dimension is : ', original_sms_df.shape)
        print('augment_sms_df dimension is : ', augment_sms_df.shape)
        
        overall_sms_df = pd.concat([original_sms_df, augment_sms_df['augment_content']], axis=1)
        print('overall sms dimension is : ', overall_sms_df.shape)
        overall_sms_df.dropna(inplace=True)
        overall_sms_df['labels'] = overall_sms_df['labels'].astype(str)
        print('After dropna overall sms dimension is : ', overall_sms_df.shape)
        return mpd.DataFrame(overall_sms_df)

    def calculate_score_bucket(self, row):
        try:
            original_text = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(row[0]))
            augment_text = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(row[1]))
            score = sentence_bleu([original_text[1:-1]], augment_text[1:-1], smoothing_function=self.smooth)
            score_bucket = math.floor(score*10)
        except Exception as e:
            print('Error: {}, Data: {}'.format(e, row))

        return score_bucket

    def run(self):
        ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True)
        overall_sms_df = self.parse_to_dataframe()
        overall_sms_df['score'] = overall_sms_df[['original_content', 'augment_content']].apply(
            self.calculate_score_bucket, axis=1)
        for score in overall_sms_df['score'].unique():
            bucket_path = os.path.join(
                self.local_bucket_augmented_sms_path, 'bucket={}'.format(score))
            if not os.path.isdir(bucket_path):
                os.makedirs(bucket_path)
            corresponding_sms_df = overall_sms_df[overall_sms_df['score'] == score][['augment_content', 'labels']]
            corresponding_sms_df.to_csv(
                os.path.join(bucket_path, 'augment_sms.csv'),
                header=False,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
                doublequote=False)
        ray.shutdown()
