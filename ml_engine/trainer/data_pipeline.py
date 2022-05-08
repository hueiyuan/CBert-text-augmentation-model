import os
import csv
import glob

import multiprocessing
from dataclasses import dataclass, field

import ray
import numpy as np
import modin.pandas as mpd
from sklearn.model_selection import train_test_split

import torch
import datasets
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Dict
from trainer.project_lib.common import utils
#from project_lib.common import utils


@dataclass(eq=False)
class SMSDatasetPipeline:
    tokenizer: PreTrainedTokenizerBase = field(
        default=None, metadata={"help": "Huggingface PreTrainedTokenizerBase object"}
    )
    augmentor_conf: dict = field(
        default=None, metadata={"help": "SMS augmentor config"}
    )

    def __post_init__(self):
        self.original_cols = self.augmentor_conf['sms_augment']['preprocess']['original_cols']
        self.character_length_limit = self.augmentor_conf['sms_augment']['preprocess']['character_length_limit']
        
        self.total_processes = self.augmentor_conf['total_processes']
        self.is_prediction = self.augmentor_conf['is_prediction']
        self.max_sequence_length = self.augmentor_conf['model_conf'].hyperparameters['max_sequence_length']
        self.batch_size = self.augmentor_conf['model_conf'].hyperparameters['batch_size']
        self.is_multiclass = bool(
            int(self.augmentor_conf['sms_augment']['preprocess']['is_multiclass']))
        self.local_processed_sms_path = self.augmentor_conf[
            'sms_augment']['local']['processed_data_folder']
        self.local_source_sms_path = self.augmentor_conf['sms_augment']['local']['source_data_folder']

        self.local_train_sms_path = os.path.join(
            self.local_processed_sms_path, 'train_sms.csv')
        self.local_test_sms_path = os.path.join(
            self.local_processed_sms_path, 'test_sms.csv')
        self.local_overall_sms_path = os.path.join(
            self.local_processed_sms_path, 'overall_sms.csv')

        self.gcs_source_sms_path = os.path.join(
            self.augmentor_conf['gcp']['ml_bucket'],
            self.augmentor_conf['sms_augment'][self.augmentor_conf['env']]['raw_sms_folder'],
            self.augmentor_conf['last_weekday_path'],
            'sms_label_data'
        )
        self.gcs_processed_datasets_sms_path = os.path.join(
            self.augmentor_conf['gcp']['ml_bucket'],
            self.augmentor_conf['sms_augment'][self.augmentor_conf['env']]['processed_sms_folder'],
            self.augmentor_conf['execute_date_path']
        )

        for folder_path in [self.local_source_sms_path, self.local_processed_sms_path]:
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

    def _mask_tokens(self,
                     mlm_probability: float,
                     inputs: torch.Tensor,
                     special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            reference: transformers.DataCollatorForLanguageModeling.mask_tokens()
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                list(
                    map(
                        lambda x: 1 if x in [
                            self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, self.tokenizer.pad_token_id] else 0, val
                    )
                ) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _convert_segment_ids(self, attention_mask, labels):
        total_token_type_ids = []
        for idx in range(len(attention_mask)):
            sentence_length = len(
                [i for i, e in enumerate(attention_mask[idx]) if e != 0])
            total_token_type_ids.append([labels[idx] for _ in range(
                sentence_length)] + [0]*(self.max_sequence_length-sentence_length))

        return total_token_type_ids

    def bert_token_preprocess(self, content, labels):
        inputs_tensor = self.tokenizer(content,
                                       return_tensors="pt",
                                       padding='max_length',
                                       max_length=self.max_sequence_length,
                                       truncation=True)
        init_ids = inputs_tensor['input_ids'].tolist()

        masked_input_ids, output_mask_labels = self._mask_tokens(mlm_probability=0.15,
                                                                 inputs=inputs_tensor['input_ids'])

        total_token_type_ids = self._convert_segment_ids(
            inputs_tensor['attention_mask'].tolist(), labels)

        tokenized_info = {
            'content': content,
            'original_labels': labels,
            'init_ids': init_ids,
            'attention_mask': inputs_tensor['attention_mask'].tolist(),
            'token_type_ids': total_token_type_ids,
            'input_ids': masked_input_ids.tolist(),
            'labels': output_mask_labels.tolist()
        }
        return tokenized_info

    def save_to_disk(self, sms_df, local_path):
        sms_df['length'] = sms_df['content'].apply(lambda value: len(value))
        sms_df = sms_df[sms_df['length'] > self.character_length_limit]
        sms_df.sort_values(by=['length'], inplace=True)
        sms_df.reset_index(drop=True, inplace=True)
        sms_df = sms_df[['content', 'labels']]
        sms_df.to_csv(local_path,
                      index=False,
                      header=False,
                      quoting=csv.QUOTE_NONNUMERIC,
                      escapechar="\\",
                      doublequote=False)

    def download_raw_sms(self):
        utils.transfer_data_by_gsutil(
            os.path.join(self.gcs_source_sms_path, '*'),
            self.local_source_sms_path)
        ray.init(num_cpus=multiprocessing.cpu_count(),
                 ignore_reinit_error=True)

        raw_sms_files = glob.glob(os.path.join(
            self.local_source_sms_path, '*.csv'))
        all_files_list = [mpd.read_csv(sms_csv) for sms_csv in raw_sms_files]
        sms_df = mpd.concat(all_files_list, axis=0, ignore_index=True)
        sms_df.dropna(inplace=True)
        sms_df['labels'] = sms_df['labels'].astype(int)

        if not self.is_prediction:
            train_sms_df, test_sms_df = train_test_split(
                sms_df, test_size=0.2, shuffle=True, stratify=sms_df.labels)

            for sms_data, sms_data_path in zip([train_sms_df, test_sms_df], [self.local_train_sms_path, self.local_test_sms_path]):
                self.save_to_disk(
                    sms_df=sms_data,
                    local_path=sms_data_path)
        else:
            self.save_to_disk(
                sms_df=sms_df,
                local_path=self.local_overall_sms_path)

        ray.shutdown()

    def load_sms_dataloader(self, dataset_type, corresponding_sms_local_path, corresponding_remove_cols):
        corresponding_sms_dataset = datasets.load_dataset('csv',
                                                          data_files=[corresponding_sms_local_path],
                                                          column_names=['content', 'labels'])
        corresponding_sms_dataset = corresponding_sms_dataset['train']
        processed_sms_dataset = corresponding_sms_dataset.map(
            self.bert_token_preprocess,
            input_columns=self.original_cols,
            num_proc=self.total_processes,
            batched=True)

        processed_sms_dataset.remove_columns_(corresponding_remove_cols)
        print('[INFO] The {} datasets length is {}'.format(
            dataset_type, len(processed_sms_dataset)))

        processed_sms_dataset.set_format(type='torch',
                                         columns=processed_sms_dataset.column_names)

        torch_sms_dataloader = torch.utils.data.DataLoader(
            processed_sms_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.total_processes)

        return torch_sms_dataloader

    def run(self):
        print('Download and preprocess sms data...')
        self.download_raw_sms()

        if not self.is_prediction:
            remove_cols = ['init_ids', 'content', 'original_labels']
            train_sms_dataloader = self.load_sms_dataloader(
                dataset_type='train',
                corresponding_sms_local_path=self.local_train_sms_path,
                corresponding_remove_cols=remove_cols)

            test_sms_dataloader = self.load_sms_dataloader(
                dataset_type='test',
                corresponding_sms_local_path=self.local_test_sms_path,
                corresponding_remove_cols=remove_cols)

            return train_sms_dataloader, test_sms_dataloader

        else:
            remove_cols = ['input_ids', 'content', 'original_labels', 'labels']
            overall_sms_dataloader = self.load_sms_dataloader(
                dataset_type='overall',
                corresponding_sms_local_path=self.local_overall_sms_path,
                corresponding_remove_cols=remove_cols)

            return overall_sms_dataloader

    def get_config(self):
        config = {
            'local_source_sms_path': self.local_source_sms_path,
            'local_processed_sms_path': self.local_processed_sms_path,
            'gcs_source_sms_path': self.gcs_source_sms_path,
        }
        return config
