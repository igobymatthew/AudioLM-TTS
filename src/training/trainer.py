import mlflow
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TokenizedTTSDataset
from src.models.vocab import CombinedVocab
from src.training.checkpointing import save_checkpoint
from src.training.optim import build_optimizer


class Trainer:
    def __init__(self, model, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = cfg.get('device', 'cpu')
        self.model.to(self.device)

        train_cfg = cfg['training']
        self.batch_size = train_cfg.get('micro_batch_size', 1)
        self.grad_accum = train_cfg.get('grad_accum_steps', 1)
        self.max_steps = train_cfg.get('max_steps', 100)
        self.log_every = train_cfg.get('log_every', 10)
        self.save_every = train_cfg.get('save_every', 100)
        self.seq_len = train_cfg.get('seq_len', 128)

        self.vocab = CombinedVocab()
        self.optimizer = build_optimizer(model, float(train_cfg.get('lr', 3e-4)))

    def _collate(self, batch):
        eos = self.vocab.text_tokenizer.eos_token_id
        input_ids = []
        for x in batch:
            text_ids = self.vocab.encode_text(x['text'])
            audio_ids = self.vocab.encode_audio(x['audio_tokens'].tolist())
            ids = (text_ids + audio_ids)[: self.seq_len]
            input_ids.append(torch.tensor(ids, dtype=torch.long))
        padded = pad_sequence(input_ids, batch_first=True, padding_value=eos)
        return padded

    def train(self, manifest_path: str):
        ds = TokenizedTTSDataset(manifest_path)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate)
        run_name = self.cfg.get('run_name', 'train_run')
        mlflow.set_experiment('audiolm_tts')

        step = 0
        self.model.train()
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({'batch_size': self.batch_size, 'max_steps': self.max_steps})
            pbar = tqdm(total=self.max_steps)
            while step < self.max_steps:
                for batch in dl:
                    batch = batch.to(self.device)
                    outputs = self.model(input_ids=batch, labels=batch)
                    loss = outputs.loss / self.grad_accum
                    loss.backward()

                    if (step + 1) % self.grad_accum == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    if step % self.log_every == 0:
                        mlflow.log_metric('loss', float(loss.item() * self.grad_accum), step=step)
                    if step > 0 and step % self.save_every == 0:
                        save_checkpoint(self.model, self.optimizer, step, self.cfg['paths']['checkpoints_dir'])

                    step += 1
                    pbar.update(1)
                    if step >= self.max_steps:
                        break
            pbar.close()
            save_checkpoint(self.model, self.optimizer, step, self.cfg['paths']['checkpoints_dir'])
