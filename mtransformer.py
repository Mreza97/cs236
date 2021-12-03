
import os
import torch
import torch.optim as optim
import tensorflow as tf
import tensorflow_text as tftext
from tqdm import tqdm
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import T5Tokenizer, T5Model, T5Config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics as tm
import torch_edit_distance

from maestrodata import *


class LevenshteinMetric(tm.Metric):
    def __init__(self, config, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("d", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.space = torch.ones(0, dtype=torch.int64)
        self.separator = torch.ones(0, dtype=torch.int64)
        self.last_value = None
        self.config = config

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"preds {preds.shape} not matching target {target.shape}"
        assert len(preds.shape) == 2, f"len(preds.shape) not 2 (is {len(preds.shape)}"

        y_len = (preds==self.config.target_eos_id).to(torch.int64)
        y_len = torch.where(y_len.any(1), y_len.argmax(1), y_len.shape[1])
        g_len = (target==self.config.target_eos_id).to(torch.int64)
        g_len = torch.where(g_len.any(1), g_len.argmax(1), g_len.shape[1])
        #print(y_len, g_len)

        y_len = y_len.int().to(preds.device)
        g_len = g_len.int().to(preds.device)
        # [ [ DEL INS SUB G-LEN ], [ .... ] ]
        dist = torch_edit_distance.levenshtein_distance(preds, target, y_len, g_len, self.space.to(preds.device), self.separator.to(preds.device))
        dist2 = dist[:, :3].sum(1)/dist[:, 3]

        self.d += dist2.sum().cpu()
        self.total += dist2.shape[0]
        self.last_value = dist

    def compute(self):
        return self.d.float() / self.total

import torchmetrics as tm

class DurationMSE(tm.Metric):
    def __init__(self, config, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.config = config
        self.add_state("duration_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def time_steps(self, x):
        where = (x >= self.config.time_event_min) & (x <= self.config.time_event_max)
        time_steps = torch.where(where,
                                 (x-self.config.time_event_min)*x.new_full((1,), 10, dtype=torch.float),
                                 x.new_full((1,), 0, dtype=torch.float)) # in msec
        return time_steps

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_time = self.time_steps(preds).sum(1)
        target_time = self.time_steps(target).sum(1)
        mse = torch.norm((preds_time - target_time), p=2, dim=0).sum(0)
        self.duration_mse += mse
        self.count += preds.shape[0]

    def compute(self):
        return self.duration_mse / self.count

import scipy

class NoteCountMSE(tm.Metric):
    def __init__(self, config, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.config = config
        self.add_state("note_count_mse", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def notes_row(self, x, counts):
        for v,c in zip(*np.unique(x, return_counts=True)):
            if v>=self.config.pitch_event_min and v<=self.config.pitch_event_max:
                counts[v - self.config.pitch_event_min] = c

    def notes(self, x):
        x = x.cpu().numpy().astype(np.int32)
        count = np.zeros((x.shape[0], self.config.pitch_event_max-self.config.pitch_event_min+1), dtype=np.int32)
        for i in range(x.shape[0]):
            self.notes_row(x[i], count[i])
        return count

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_sz = preds.shape[0]
        p = self.notes(preds)
        t = self.notes(target)
        self.note_count_mse += np.linalg.norm(p-t, ord=2, axis=1).sum(0)
        self.count += batch_sz

    def compute(self):
        return self.note_count_mse / self.count


class Model(LightningModule):
    def __init__(self, config, **params):
        self.config = config
        self.params = params
        self.metrics = ['duration_ms_mse', 'note_count_mse']
        self.duration_ms_mse = DurationMSE(self.config)
        self.note_count_mse = NoteCountMSE(self.config)
        super().__init__()
        self.t5 = T5Model.from_pretrained("t5-small")
        # do we want linear without bias here? why not with activation.
        #self.lm_head = torch.nn.Linear(self.t5.config.d_model, self.t5.config.vocab_size, bias=False)
        self.lm_head = torch.nn.Linear(self.t5.config.d_model, self.config.vocab_size, bias=False)
        # config.d_model is the embedding size (typically 512)
        #self.input_embeddings = torch.nn.Embedding(maestro_config.vocab_size, self.t5.config.d_model)
        self.encoder_input = torch.nn.Linear(self.config.n_mels, self.t5.config.d_model)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.label_padding_id)
        if torch.cuda.is_available():
            self.levenshtein = LevenshteinMetric(self.config)

    def forward(self, x):
        inputs_embeds = self.encoder_input(x['mfcc'])

        # forward + backward + optimize
        #inputs_embeds = input_embeddings(input_ids).to(device)
        ret = self.t5(inputs_embeds=inputs_embeds, decoder_attention_mask=x['attention_mask'], decoder_input_ids=x['target_id'], return_dict=True)
        lm_logits = self.lm_head(ret.last_hidden_state)
        #print(f"forward() - target {x['target_id']}")
        #print(f"forward() - logits {lm_logits[0].argmax(1)}")
        return lm_logits

    def training_step(self, x, batch_idx):
        lm_logits = self(x)
        preds = lm_logits.view(-1, lm_logits.size(-1))
        target = x['label'].view(-1)
        #print("PREDS", preds.dtype, preds.shape, preds)
        #print("TARGET", target.dtype, target.shape, target.view(-1))
        #
        # loss here is CE loss between class probabilities (float32) and target (int64)
        #    pred dimensions [batch, max_target_length+1, vocab_size]
        #    target dimensions [ batch ]
        #
        loss = self.criterion(preds, target)
        #self.log("training_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step_end(self, step_output):
        loss = step_output.mean()
        self.log("training_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #def validation_step2(self, x, batch_idx):
    #    lm_logits = self(x)
    #    loss = self.criterion(lm_logits.view(-1, lm_logits.size(-1)), x['label'].view(-1))
    #    self.log("validation_loss", loss, on_step=False, on_epoch=True)
    #    return loss

    def validation_step(self, x, batch_idx):
        lm_logits = self.generate(x, batch_idx)
        preds = lm_logits.view(-1, lm_logits.size(-1))
        target = x['label']#.view(-1)
        if self.levenshtein:
            self.levenshtein(preds, target)
            self.log('levenshtein', self.levenshtein, on_step=True, on_epoch=True)
        for k in self.metrics:
            m = getattr(self, k)
            m(preds, target)
            print("LOGGING", k, m)
            self.log(k, m, on_step=True, on_epoch=True)
        return None

    #def validation_step_end(self, step_output):
    #    for k,m in self.metrics.items():
    #        self.log(k, m.compute())
    #    if self.levenshtein:
    #        dist = self.levenshtein.compute()
    #        self.log("levenshtein", dist, on_step=False, on_epoch=True)
    #        last_value = self.levenshtein.last_value.to(torch.float)
    #        self.log("levenshtein_del", last_value[:, 0].mean(), on_step=True, on_epoch=False)
    #        self.log("levenshtein_ins", last_value[:, 1].mean(), on_step=True, on_epoch=False)
    #        self.log("levenshtein_sub", last_value[:, 2].mean(), on_step=True, on_epoch=False)
    #        self.log("levenshtein_glen", last_value[:, 3].mean(), on_step=True, on_epoch=False)
    #    return None

    def generate(self, x, batch_idx, use_cache=True, tqdm=None, stop_at_eos=True):
        input_ids = x['mfcc']
        inputs_embeds = self.encoder_input(input_ids)
        batch_size = input_ids.shape[0]
        target_ids = input_ids.new_full((batch_size,self.config.max_target_length+1,), self.config.target_padding_id, dtype=torch.int64)
        attention_mask = input_ids.new_full((batch_size,self.config.max_target_length+1,), self.config.mask_padding_id, dtype=torch.int64)
        #predictions = input_ids.new_full((batch_size,self.config.max_target_length+1,), self.config.target_padding_id, dtype=torch.float32)

        target_ids[:,0] = self.config.target_bos_id
        last_target_ids = input_ids.new_full((batch_size, 1), self.config.target_bos_id, dtype=torch.int64)
        attention_mask[:,0] = self.config.mask_id
        completed = 0
        past_key_values = None
        device = self.device
        
        # we should not have loops of varying sizes when using pytorch & TPUs (see xla)
        #while completed<batch_size and target_seq_length<10: #self.config.max_target_length:
        r = range(1, self.config.max_target_length)
        if tqdm: r = tqdm(r, total=self.config.max_target_length-1)
        for target_seq_length in r:
            torch.cuda.empty_cache() 
            ret = self.t5(inputs_embeds=inputs_embeds,
                          decoder_attention_mask=attention_mask if (past_key_values is None or not use_cache) else None,
                          decoder_input_ids=target_ids if (past_key_values is None or not use_cache) else last_target_ids,
                          return_dict=True,
                          past_key_values=past_key_values if use_cache else None,
                          use_cache=(target_seq_length>1 and use_cache))

            lm_logits = self.lm_head(ret.last_hidden_state).cpu()
            past_key_values = ret.past_key_values

            target_dim = 0 if lm_logits.shape[1]==1 else target_seq_length-1
            next_target_ids = lm_logits[:, target_dim].argmax(1)
            completed = completed + (next_target_ids==self.config.target_eos_id).sum()
            target_ids[:, target_seq_length] = next_target_ids
            #predictions[:, target_seq_length] = lm_logits[:, target_dim, next_target_ids]
            last_target_ids[:, 0] = next_target_ids
            attention_mask[:, target_seq_length] = self.config.mask_id
            if completed == batch_size and stop_at_eos: break

        target_ids[:, 0:-1] = target_ids[:, 1:].clone()
        target_ids[: -1] = self.config.label_padding_id

        return target_ids #, predictions

    #https://discuss.huggingface.co/t/t5-finetuning-tips/684/4
    def configure_optimizers(self):
        lr = self.params['lr'] if 'lr' in self.params else 1e-3

        return Adafactor(
            self.parameters(),
            lr=lr, #1e-3 ... 1e-4
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )


