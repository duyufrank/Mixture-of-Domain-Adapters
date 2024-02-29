import os
from functools import partial
from glob import glob

import pytorch_lightning as pl
import torch
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice, PretrainedConfig, AutoTokenizer, AutoModelForMultipleChoice
from argparse import ArgumentParser
from torchmetrics import F1Score, Accuracy, BLEUScore # newly add BLEUScore
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import wandb
from transformers.adapters.configuration import HoulsbyConfig, PfeifferConfig
import sys
from src.models.transformers import LlamaForCausalLM, LlamaConfig
from transformers.adapters import PrefixTuningConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
# from opendelta import LoraModel
from pytorch_lightning.strategies.ddp import DDPStrategy
import random

NUM_CHOICES = 5

def preprocess(data, tokenizer, max_length, trim=False):
    q = data["question"]
    rep_q = [item for item in q for _ in range(5)]
    c = data["choices"]
    expanded_c = [e for ele in c for e in ele["text"]]
    x = tokenizer(rep_q, expanded_c, return_tensors='pt', padding='max_length', truncation=True,
                  max_length=max_length).data
    end = 200 if trim else len(x["input_ids"])
    x = {k: v.view(-1, NUM_CHOICES, max_length)[:end] for k, v in x.items()}
    y = data["answerKey"][:end]
    y = torch.tensor([ord(item) - ord("A") for item in y])

    return x, y


class DictDataset(Dataset):
    def __init__(self, x, y, few_shot=None):
        self.x = x
        self.y = y
        if few_shot is not None:
            new_results = self.preprocess_function_k_shot(
                {'x': self.x, 'label': self.y}, few_shot
            )
            self.x = new_results['x']
            self.y = new_results['label']

    def __len__(self):
        return len(self.x["input_ids"])

    def __getitem__(self, idx):
        sample = {key: self.x[key][idx] for key in self.x.keys()}
        sample["labels"] = self.y[idx]
        return sample
    
    @staticmethod
    def preprocess_function_k_shot(examples, few_shot):
        random_indices = list(range(0, len(examples["label"])))
        random.shuffle(random_indices)

        new_examples = {'x': {}, 'label': []}
        for key in examples['x'].keys():
            new_examples['x'][key] = []
        label_count = {}

        for index in random_indices:      
            label = int(examples['label'][index])
            if label not in label_count:
                label_count[label] = 0

            if label_count[label] < few_shot:
                for key in examples['x'].keys():
                    new_examples['x'][key].append(examples['x'][key][index])
                new_examples['label'].append(examples['label'][index])
                label_count[label] += 1
        
        for key in examples['x'].keys():
            new_examples['x'][key] = torch.stack(new_examples['x'][key], dim=0)
        new_examples['label'] = torch.Tensor(new_examples['label']).long()
        
        print('k-shot selection done!!')
        
        return new_examples
    
class ConvDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, maxlen=None, src_fields=["question", "description"], tgt_field="answer"):
        self.dataset = dataset
        self.src_fields = src_fields
        self.tgt_field = tgt_field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]

        text = '问题: '

        for field in self.src_fields:
            text += data[field] + ' '

        text += '回答: '+data[self.tgt_field]+'</s>'

        return {'text': text}
    
    def collate_fn(self, batch):
        x_encoding = self.tokenizer(
            [x["text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.maxlen,
            return_tensors="pt",
        ).data

        batched_data = {
            'input_ids': x_encoding["input_ids"],
            'attention_mask': x_encoding["attention_mask"],
            'labels': x_encoding["input_ids"],
        }
        return batched_data

        


def acc_from_logits_and_labels(logits, labels, accuracy_fn):
    acc = accuracy_fn(logits, labels)
    return acc


class Model(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=3)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=32)

        parser.add_argument("--model_name", type=str,
                            default="/data/weiguang/Llama2-Chinese-7b-Chat")

        parser.add_argument('--load_adapters', type=str)
        parser.add_argument('--load_task_adapter', type=str)
        parser.add_argument('--adapter_type', type=str)
        parser.add_argument('--adapter_non_linearity', type=str, default='swish')
        parser.add_argument('--layers', type=str)
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=8)
        parser.add_argument('--disable_moe', action='store_true')
        parser.add_argument(
            "--dirpath", type=str, default='results/csqa'
        )
        parser.add_argument('--few_shot', type=int)
        return parser
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.layers = [] if self.hparams.layers is None or self.hparams.layers == '' else [int(x) for x in self.hparams.layers.split(',')]
        self.conv_dataset_raw = load_dataset('csv', data_dir="/home/weiguang/SaBART/matinf_1.0_encrypted", data_files={'train': 'train.csv', 'val': 'dev.csv', 'test': 'test.csv'})
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        # training_data = dataset["train"]
        # self.x_train, self.y_train = preprocess(training_data, tokenizer, 1024)
        # self.x_val, self.y_val = preprocess(dataset["validation"], tokenizer, 1024)
        self.ka_list = [] if self.hparams.load_adapters is None or self.hparams.load_adapters == '' else [x for x in self.hparams.load_adapters.split(',')]

        model_config = LlamaConfig.from_pretrained(
            self.hparams.model_name,
            layers=self.hparams.layers,
            output_original=True,
            #num_labels=NUM_CHOICES,
            enable_old_ka=True,
            num_of_kas=len(self.ka_list) if self.ka_list is not None else 0,
            disable_moe=self.hparams.disable_moe,
        )
        model_config.max_position_embeddings = 1024
        self.model = LlamaForCausalLM.from_pretrained(self.hparams.model_name, config=model_config).to(self.device)
        if self.ka_list is not None and len(self.ka_list) > 0:
            self.model.load_knowledge_adapter(self.ka_list)
            print('adapter loaded!!!')
        self.bleu_metric = BLEUScore()
        # self.f1 = F1Score(num_classes=NUM_CHOICES, average='macro')
        self.batch_size = self.hparams.batch_size
        
        if self.hparams.load_task_adapter is not None:
            self.model.load_adapter(self.hparams.load_task_adapter)
            self.model.train_adapter('domain', 'pfeiffer')
        elif self.hparams.adapter_type == 'pfeiffer':
            adapter_config = PfeifferConfig(
                ln_after=False,
                ln_before=False,
                mh_adapter=False,
                output_adapter=True,
                adapter_residual_before_ln=False,
                non_linearity=self.hparams.adapter_non_linearity,
                original_ln_after=True,
                original_ln_before=True,
                reduction_factor=16,
                residual_before_ln=True
            )
        elif self.hparams.adapter_type == 'houlsby':
            adapter_config = HoulsbyConfig(
                ln_after=False,
                ln_before=False,
                mh_adapter=True,
                output_adapter=True,
                adapter_residual_before_ln=False,
                non_linearity=self.hparams.adapter_non_linearity,
                original_ln_after=True,
                original_ln_before=False,
                reduction_factor=16,
                residual_before_ln=True
            )
        elif self.hparams.adapter_type == 'lora':
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.hparams.lora_r, lora_alpha=self.hparams.lora_alpha, lora_dropout=0.1)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        elif self.hparams.adapter_type == 'prefix_tuning':
            adapter_config = PrefixTuningConfig(flat=False, prefix_length=30)
        if self.hparams.adapter_type != 'lora' and self.hparams.adapter_type is not None and self.hparams.load_task_adapter is None:
            print('Using no adapter')
            self.model.add_adapter('domain', self.hparams.adapter_type, adapter_config)
            self.model.train_adapter('domain', self.hparams.adapter_type)

        # We train MOE gate parameters!
        if not self.hparams.disable_moe:
            for layer in self.hparams.layers:
                moe_gate = self.model.model.model.layers[layer].mlp.gating
                for param in moe_gate.parameters():
                    param.requires_grad_(True)
        
        print('=' * 50)
        print('The following parameters are not frozen:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        print('=' * 50)
        
        self.best_metrics = { 'bleu': 0.0 } #{ 'acc': 0.0 }

    def log_each_step(self, name, val, on_step=True, on_epoch=True):
        self.log(name, val, prog_bar=True, on_step=on_step, on_epoch=on_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        train_dataset = ConvDataset(
                self.conv_dataset_raw['train'],
                # concatenate_datasets(
                #     [self.conv_dataset_raw['train'].select(range(720000, len(self.conv_dataset_raw['train']))),
                #      self.conv_dataset_raw['train'].select(range(720000))]),
                self.tokenizer, 
                maxlen=self.model.config.max_position_embeddings
            )
        return DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=train_dataset.collate_fn)
        #return DataLoader(DictDataset(self.x_train, self.y_train, few_shot=self.hparams.few_shot), batch_size=self.batch_size)

    def val_dataloader(self):
        val_dataset = ConvDataset(
                self.conv_dataset_raw['val'], 
                self.tokenizer, 
                maxlen=self.model.config.max_position_embeddings
            )
        return DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=val_dataset.collate_fn)
        #return DataLoader(DictDataset(self.x_val, self.y_val, few_shot=self.hparams.few_shot), batch_size=self.batch_size)

    def forward(self, **x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(**x)
        return outputs

    def training_step(self, batch, batch_idx):
        self.log('monitoring_step', self.global_step)
        labels = batch["labels"]
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log_each_step(f'train_loss', loss)
        predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in torch.argmax(logits, dim=-1)]
        bleu = self.bleu_metric(predictions, [[self.tokenizer.decode(label, skip_special_tokens=True)] for label in labels])
        # acc = acc_from_logits_and_labels(logits, labels, self.accuracy)
        # f1 = acc_from_logits_and_labels(logits, labels, self.f1)
        self.log_each_step(f'train_bleu', bleu)
        # self.log_each_step(f'train_acc', acc)
        #self.log_each_step(f'train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        # reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
        predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in torch.argmax(logits, dim=-1)]
        bleu = self.bleu_metric(predictions, [[self.tokenizer.decode(label, skip_special_tokens=True)] for label in labels])
        # acc = acc_from_logits_and_labels(reshaped_logits, batch["labels"], self.accuracy)
        # f1 = acc_from_logits_and_labels(reshaped_logits, batch["labels"], self.f1)
        self.log_each_step(f'val_step_bleu', bleu)
        # self.log_each_step(f"val_step_acc", acc)
        # self.log_each_step(f"val_step_f1", f1)
        self.log_each_step(f"val_step_loss", loss)

        return {
            "val_step_loss": loss,
            'val_step_bleu': bleu,
            # "val_step_acc": acc,
            # "val_step_f1": f1,
        }

    def validation_epoch_end(self, outputs):
        print("\n\nvalidation_epoch_end\n\n")
        avg_loss = torch.tensor(
            [x["val_step_loss"] for x in outputs]).mean()
        avg_bleu= torch.tensor(
            [x["val_step_bleu"] for x in outputs]).mean()
        # avg_acc = torch.tensor(
        #     [x["val_step_acc"] for x in outputs]).mean()
        # avg_f1 = torch.tensor(
        #     [x["val_step_f1"] for x in outputs]).mean()
        self.log_each_step("val_epoch_loss", avg_loss, on_step=False)
        self.log_each_step("val_epoch_bleu", avg_bleu, on_step=False)
        # self.log_each_step("val_epoch_acc", avg_acc, on_step=False)
        # self.log_each_step("val_epoch_f1", avg_f1, on_step=False)
        if avg_bleu > self.best_metrics['bleu']:
            self.best_metrics['bleu'] = avg_bleu
        # if avg_acc > self.best_metrics['acc']:
        #     self.best_metrics['acc'] = avg_acc
        # if avg_f1 > self.best_metrics['f1']:
        #     self.best_metrics['f1'] = avg_f1
        
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int)
    parser.add_argument("--project_name", type=str, default="add-fever-complete")
    parser.add_argument("--run_name", type=str, default="test")

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        seed_everything(seed=args.seed)

    logger = WandbLogger(project=args.project_name, name=args.run_name)

    callbacks = [
        LearningRateMonitor(
            logging_interval="step",
        ),
        ModelCheckpoint(
            dirpath=args.dirpath,
            filename="{epoch}-{step}",
            every_n_train_steps=5000,  # Save a checkpoint every 1000 training steps
            save_top_k=3,
            mode='max',
            monitor='monitoring_step',
        ),
    ]

    trainer = Trainer.from_argparse_args(
        args, 
        precision=16, #newly add
        logger=logger, 
        callbacks=callbacks,
        strategy='ddp',
        num_sanity_val_steps=0
    )

    model = Model(**vars(args))
    

    trainer.fit(model) #, ckpt_path='/home/weiguang/Mixture-of-Domain-Adapters/results/cls_finetune_mixda_seed42_bos_notn/epoch=0-step=20000.ckpt')
    
    for m in ('bleu'): #, 'acc','f1'):
        print("#####" + m + '####:' + str(model.best_metrics[m]))
