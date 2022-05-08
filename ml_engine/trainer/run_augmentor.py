#!/usr/bin/env python3
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"  # ddp test
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # ddp test
import io
import csv
import yaml
import time
import warnings
import pendulum
from typing import List
from absl import app, flags, logging
from dataclasses import dataclass, field

import wandb
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler, autocast

## transformers lib
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizer, BertConfig
from datasets.utils.logging import set_verbosity_error

## custom lib
from trainer.custom_lib import sms_augmented_model_config
from trainer.project_lib.common import loadconfig
from trainer.project_lib.common import utils
from trainer.custom_lib import tokenize_utils
from trainer.data_pipeline import SMSDatasetPipeline
from trainer.model import CbertModel
from trainer.optimization import CbertOptimizer
from trainer.bleuscore import BleuScoreMetrics
from trainer.callbacks import AugmentCallbacks

'''
from project_lib.common import loadconfig
from project_lib.common import utils
from custom_lib import tokenize_utils
from data_pipeline import SMSDatasetPipeline
from model import CbertModel, SMSAugmentorPLModel
from optimization import CbertOptimizer
from wandb_logger import SMSAugmentorLogger
from callbacks import SMSCallbacks
from bleuscore import BleuScoreMetrics
'''
set_verbosity_error()  # disable progress bar

FLAGS = flags.FLAGS
flags.DEFINE_string('job-dir', default=None,
                    help='GCS or local dir to write checkpoints and export model.')
flags.DEFINE_string('exec_date', default=None, 
                    help='GCP ml-engine job execution date.')
flags.DEFINE_string('last_weekday_date', default=None, 
                    help='The last week day for cleaned data')
flags.DEFINE_string('country', default=None, 
                    help='The country of augment sms')
flags.DEFINE_string('env_type', default=None, 
                    help='The environment value. production: serve; stage: serve-staging')
flags.DEFINE_string('wandb_apikey', default=None, 
                    help='The apikey for weight and biases access')
flags.DEFINE_string('is_prediction', default=None, 
                    help='The task job just for predict sms data')
flags.mark_flags_as_required(
    ['job-dir', 'exec_date', 'last_weekday_date', 'country', 'wandb_apikey', 'env_type', 'is_prediction'])


def generate_augmentor_config(flags_conf):
    general_config = loadconfig.config(
        service_name='sms-augmentation',
        platform='gcp',
        config_folder_prefix='project_config',
        config_prefix_name='general')._load()

    project_config = loadconfig.config(
        service_name='sms-augmentation',
        platform='gcp',
        config_folder_prefix='project_config',
        config_prefix_name='sms-augmentation')._load()
    
    model_config = sms_augmented_model_config.ModelConfig()
    
    print('[INFO] Execute data is {0}. Last weekday is {1}'.format(
        flags_conf.exec_date, flags_conf.last_weekday_date))
    print('[INFO] This job work on {0}'.format(flags_conf.env_type))

    execute_date_path = 'year={year}/month={month}/day={day}/country={country}'.format(
        year=pendulum.parse(flags_conf.exec_date).year,
        month=pendulum.parse(flags_conf.exec_date).strftime('%m'),
        day=pendulum.parse(flags_conf.exec_date).strftime('%d'),
        country=flags_conf.country)

    last_weekday_path = 'year={year}/month={month}/day={day}/country={country}'.format(
        year=pendulum.parse(flags_conf.last_weekday_date).year,
        month=pendulum.parse(flags_conf.last_weekday_date).strftime('%m'),
        day=pendulum.parse(flags_conf.last_weekday_date).strftime('%d'),
        country=flags_conf.country)

    env_type = flags_conf.env_type
    exec_date = flags_conf.exec_date
    is_prediction = bool(int(flags_conf.is_prediction))
    
    augmentor_config = {
        **general_config, 
        **project_config,
        'model_conf': model_config,
        'env':env_type,
        'exec_date':exec_date,
        'execute_date_path':execute_date_path,
        'last_weekday_path':last_weekday_path,
        'is_prediction':is_prediction,
        'country':flags_conf.country,
        'wandb_apikey': flags_conf.wandb_apikey,
        'total_processes': os.cpu_count()
    }
    print('[INFO] Generate augmentor_config done.')
    return augmentor_config

def generate_bert_tokenizer(pretrained):
    bert_config = BertConfig.from_pretrained(pretrained)
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained, config=bert_config)

    return bert_tokenizer

def initial_wandb_service(augmentor_conf):
    print('[INFO] Initialize weight & bias service.')
    time_tag = pendulum.now(tz='Asia/Taipei').strftime('%Y%m%d-%H%M%S')
    wandb.login(key=augmentor_conf['wandb_apikey'], relogin=False)
    wandb_log_conf = {
        **augmentor_conf['model_conf'].training_parameters,
        **augmentor_conf['model_conf'].hyperparameters,
        'learning_rate': augmentor_conf['model_conf'].optimizer_parameters['learning_rate']
    }
    
    init_wandb = wandb.init(
        project='{}-{}'.format(augmentor_conf['env'], augmentor_conf['project_name']),
        group=augmentor_conf['env'],
        job_type='training',
        name='{}_{}'.format(augmentor_conf['project_name'], time_tag),
        config=wandb_log_conf,
        tags=[
            augmentor_conf['project_name'],
            augmentor_conf['env'],
            time_tag
        ],
        sync_tensorboard=False,
        reinit=True
    )
    return init_wandb


@dataclass(eq=False)
class SMSAugmentor:
    tokenizer: PreTrainedTokenizerBase = field(
        default=None, metadata={"help": "Huggingface PreTrainedTokenizerBase"}
    )
    train_dataloader: DataLoader = field(
        default=None, metadata={"help": "torch dataloader object for training dataset"}
    )
    test_dataloader: DataLoader = field(
        default=None, metadata={"help": "torch dataloader object for test dataset"}
    )
    augmentor_conf: dict = field(
        default=None, metadata={"help": "sms augmentor config"}
    )

    def __post_init__(self):
        self.num_labels = self.augmentor_conf['model_conf'].training_parameters['num_of_labels']
        self.epochs = self.augmentor_conf['model_conf'].training_parameters['epochs']

        self.num_warmup_steps = int(
            len(self.train_dataloader) * self.epochs * 0.1)
        self.num_training_steps = (
            len(self.train_dataloader) * self.epochs)

        self.model_name = self.augmentor_conf['model_conf'].training_parameters['model_name']
        self.local_model_path = self.augmentor_conf['sms_augment']['local']['model_path']
        self.gcs_model_path = os.path.join(
            self.augmentor_conf['gcp']['ml_bucket'],
            self.augmentor_conf['sms_augment'][self.augmentor_conf['env']]['model_folder'],
            self.augmentor_conf['execute_date_path'])

        if not os.path.isdir(self.local_model_path):
            os.makedirs(self.local_model_path)

    def initial_model(self):
        init_cbert_model = CbertModel(
            hyperparameters=self.augmentor_conf['model_conf'].hyperparameters,
            num_labels=self.num_labels)

        model = init_cbert_model.build_model()

        return model

    def initial_callbacks(self):
        overall_callbacks = AugmentCallbacks(
            callback_parameters=self.augmentor_conf['model_conf'].callback_parameters)
        
        return overall_callbacks

    def initial_optimizer(self, model):
        init_optimizer = CbertOptimizer(
            model_params=list(model.named_parameters()),
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            optimizer_parameters=self.augmentor_conf['model_conf'].optimizer_parameters
        )
        optimizer = init_optimizer.get_optimizer_and_scheduler()

        return optimizer

    def train(self, model, scaler, optimizer):
        train_loss = 0.0

        for step, batch in enumerate(self.train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            labels = batch['labels'].cuda()

            with autocast():
                outputs = model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = outputs[0].mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss/len(self.train_dataloader)
        train_info = {
            'train_loss': avg_train_loss
        }

        return model, train_info

    @torch.no_grad()
    def evaluate(self, model):
        valid_loss = 0.0

        for step, batch in enumerate(self.test_dataloader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            labels = batch['labels'].cuda()

            with autocast():
                outputs = model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = outputs[0].mean()

            valid_loss += loss.item()

        avg_valid_loss = valid_loss/len(self.test_dataloader)
        perplexity = torch.exp(torch.tensor(avg_valid_loss))

        valid_info = {
            'val_loss': avg_valid_loss,
            'perplexity': perplexity,
        }
        return model, valid_info

    def run(self):
        scaler = GradScaler()
        model = self.initial_model()
        optimizer = self.initial_optimizer(model)
        overall_callbacks = self.initial_callbacks()

        print('[INFO] Starting training.....')
        for epoch in range(self.epochs):
            print('[INFO] Currently the training epoch is {}'.format(epoch))
            start_time = time.time()
            
            model.train()
            model, train_info = self.train(model, scaler, optimizer)
            
            model.eval()
            model, valid_info = self.evaluate(model)

            wandb_logger_info = {
                **train_info,
                **valid_info,
                'avg_epoch_seconds': time.time() - start_time
            }
            wandb.log(wandb_logger_info)

            overall_callbacks(
                epoch=epoch, 
                monitor_loss=valid_info['val_loss']
            )
            if overall_callbacks.callback_stop_status:
                break

        torch.save(model, os.path.join(self.local_model_path, self.model_name))

    def upload_model(self):
        utils.transfer_data_by_gsutil(
            input_path=os.path.join(self.local_model_path, self.model_name),
            output_path=os.path.join(self.gcs_model_path, self.model_name))
        

@dataclass(eq=False)
class SMSAugmentorPredictor:
    tokenizer: PreTrainedTokenizerBase = field(
        default=None, metadata={"help": "Huggingface PreTrainedTokenizerBase"}
    )
    predict_dataloader: DataLoader = field(
        default=None, metadata={"help": "torch dataloader object for training dataset"}
    )
    augmentor_conf: dict = field(
        default=None, metadata={"help": "sms augmentor config"}
    )
    
    def __post_init__(self):
        self.original_file_path = os.path.join(
            self.augmentor_conf['sms_augment']['local']['augment_data_folder'], 
            'original_sms.csv')
        self.augment_file_path = os.path.join(
            self.augmentor_conf['sms_augment']['local']['augment_data_folder'], 
            'augment_sms.csv')
        self.gcs_augment_data_path = os.path.join(
            self.augmentor_conf['gcp']['ml_bucket'],
            self.augmentor_conf['sms_augment'][self.augmentor_conf['env']]['augmented_sms_folder'], 
            self.augmentor_conf['execute_date_path'])
    
        self.model_name = self.augmentor_conf['model_conf'].training_parameters['model_name']
        self.local_model_path = self.augmentor_conf['sms_augment']['local']['model_path']
        self.gcs_model_path = os.path.join(
            self.augmentor_conf['gcp']['ml_bucket'],
            self.augmentor_conf['sms_augment'][self.augmentor_conf['env']]['model_folder'],
            self.augmentor_conf['execute_date_path'])
        
        self.temperature_value = self.augmentor_conf['model_conf'].augmentation_parameters['temperature_value']
        self.sample_ratio = self.augmentor_conf['model_conf'].augmentation_parameters['sample_ratio']

        for local_path in [self.augmentor_conf['sms_augment']['local']['augment_data_folder'], self.local_model_path]:
            if not os.path.isdir(local_path):
                os.makedirs(local_path)
    
    def download_model(self):
        utils.transfer_data_by_gsutil(
            input_path=os.path.join(self.gcs_model_path, self.model_name),
            output_path=os.path.join(self.local_model_path, self.model_name))
        
        model = torch.load(os.path.join(
            self.local_model_path, self.model_name)).module
        
        return model
    
    @torch.no_grad()
    def prediction(self):
        print('[INFO] Initial and Download Cbert model from gcs to local.')
        trained_augmentor_model = self.download_model()
        trained_augmentor_model.eval()
        
        for step, batch in enumerate(self.predict_dataloader):
            original_sms_dict = {'content': [], 'labels': []}
            augmented_sms_dict = {'content': [], 'labels': []}

            init_ids = batch['init_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()

            input_lens = [sum(mask).item() for mask in attention_mask]
            masked_idx = np.squeeze(
                [np.random.randint(0, l, max(l//self.sample_ratio, 1)) for l in input_lens])

            for ids, seg, idx in zip(init_ids, token_type_ids, masked_idx):
                original_sms_dict['content'].append(tokenize_utils.convert_ids_to_str(ids.cpu().numpy(), self.tokenizer))
                original_sms_dict['labels'].append(str(int(seg.cpu().numpy()[0])))
                ids[idx] = 103

            outputs = trained_augmentor_model(
                init_ids, attention_mask, token_type_ids)
            predictions = torch.nn.functional.softmax(
                outputs[0]/self.temperature_value, dim=2)

            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, token_type_ids):
                preds = torch.multinomial(preds, 1, replacement=True)[idx]

                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred
                new_str = tokenize_utils.convert_ids_to_str(ids.cpu().numpy(), self.tokenizer)
                augmented_sms_dict['content'].append(new_str)
                augmented_sms_dict['labels'].append(
                    str(int(seg.cpu().numpy()[0])))

            original_sms_df = pd.DataFrame(original_sms_dict)
            original_sms_df.to_csv(self.original_file_path,
                                mode='a',
                                index=False,
                                header=False,
                                quoting=csv.QUOTE_NONNUMERIC,
                                escapechar="\\",
                                doublequote=False)
            augmented_sms_df = pd.DataFrame(augmented_sms_dict)
            augmented_sms_df.to_csv(self.augment_file_path,
                                    mode='a',
                                    index=False,
                                    header=False,
                                    quoting=csv.QUOTE_NONNUMERIC,
                                    escapechar="\\",
                                    doublequote=False)
        utils.transfer_data_by_gsutil(
            input_path=self.original_file_path,
            output_path=os.path.join(self.gcs_augment_data_path, 'original_sms.csv'))
        utils.transfer_data_by_gsutil(
            input_path=self.augment_file_path,
            output_path=os.path.join(self.gcs_augment_data_path, 'augment_sms.csv'))
        
        torch.cuda.empty_cache()
        print('[INFO] Predict and save augmented sms file done.')
    
    
def main(argv):
    augmentor_config = generate_augmentor_config(flags_conf=FLAGS)
    bert_tokenizer = generate_bert_tokenizer(
        pretrained=augmentor_config['model_conf'].hyperparameters['bert_pretrained'])
    
    if not augmentor_config['is_prediction']:
        data_pipeline = SMSDatasetPipeline(
            tokenizer=bert_tokenizer,
            augmentor_conf=augmentor_config)
        train_sms_dataloader, test_sms_dataloader = data_pipeline.run()
    
        init_wandb = initial_wandb_service(augmentor_conf=augmentor_config)
        
        print('[INFO] Execute CBert model training.')
        sms_augmentor = SMSAugmentor(
            tokenizer=bert_tokenizer,
            train_dataloader=train_sms_dataloader,
            test_dataloader=test_sms_dataloader,
            augmentor_conf=augmentor_config)

        sms_augmentor.run()
        init_wandb.finish()
        print('[INFO] Training model done.')
        
        sms_augmentor.upload_model()
        print('[INFO] Upload Cbert model to GCS done.')
        
    else:
        data_pipeline = SMSDatasetPipeline(
            tokenizer=bert_tokenizer,
            augmentor_conf=augmentor_config)
        overall_sms_dataloader = data_pipeline.run()
        
        print('[INFO] Execute CBert model prediction.')
        sms_augment_predictor = SMSAugmentorPredictor(
            tokenizer=bert_tokenizer,
            predict_dataloader=overall_sms_dataloader,
            augmentor_conf=augmentor_config)

        sms_augment_predictor.prediction()
        print('[INFO] SMS augment model prediction done.')

        local_bucket_augmented_sms_path = os.path.join(
            augmentor_config['sms_augment']['local']['augment_data_folder'],
            'buckets')

        bleu_calculator = BleuScoreMetrics(
            local_bucket_augmented_sms_path=local_bucket_augmented_sms_path,
            tokenizer=bert_tokenizer)
        bleu_calculator.run()
        print('[INFO] Execute similarity score with bleu done.')

        local_augmented_sms_bucket_path = os.path.join(
            augmentor_config['sms_augment']['local']['augment_data_folder'],
            'buckets', '*')
        gcs_augmented_sms_path = os.path.join(
            augmentor_config['gcp']['ml_bucket'],
            augmentor_config['sms_augment'][augmentor_config['env']
                                            ]['augmented_sms_folder'],
            augmentor_config['execute_date_path']
        )
        print('[INFO] Upload augment file from {} to {}.'.format(
            local_augmented_sms_bucket_path, gcs_augmented_sms_path))

        utils.transfer_data_by_gsutil(
            local_augmented_sms_bucket_path,
            gcs_augmented_sms_path, True)

        print('[INFO] Upload augment file to GCS done.')

    
if __name__ == "__main__":
    app.run(main)
