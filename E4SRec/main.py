

from typing import Dict, Tuple, Union, Optional

import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from transformers.trainer import Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()

cfg.add_argument("--saved-model", type=str, default="./models/Platypus2-7B")
cfg.add_argument("--prompt-template", type=str, default="./configs/alpaca.json")
cfg.add_argument("--maxlen", type=int, default=50)

# Finetune
cfg.add_argument("--lora-rank", type=int, default=16)
cfg.add_argument("--lora-alpha", type=int, default=16)
cfg.add_argument("--lora-dropout", type=float, default=0.05)

cfg.set_defaults(
    description="E4SRec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=3,
    batch_size=64,
    gradient_accumulation_steps=1,
    optimizer='AdamW',
    lr=3.e-4,
    weight_decay=0.,
    seed=1,
)
cfg.compile()

cfg.lora_config = LoraConfig(
    r=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    target_modules=[
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=cfg.lora_dropout,
    task_type='FEATURE_EXTRACTION',
    bias='none'
)
cfg.quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)


class E4SRec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.model = LlamaModel.from_pretrained(
            cfg.saved_model,
            quantization_config=cfg.quantization_config,
            dtype=torch.float16,
            local_files_only=True,
            device_map=cfg.device
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, cfg.lora_config)
        self.model.config.use_cache = False

        # self.model.save_pretrained()

        self.tokenizer = LlamaTokenizer.from_pretrained(
            cfg.saved_model,
            use_fast=False, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        with open(cfg.prompt_template) as f:
            prompts = json.load(f)
        instruction = prompts['prompt_input'].format(
            instruction="Given the user’s purchase history, predict next possible item to be purchased."
        )
        response = prompts['response_split']

        instruct_ids, instruct_mask = self.tokenizer(
            instruction,
            padding=False, return_tensors='pt', 
            add_special_tokens=False
        ).values()
        response_ids, response_mask = self.tokenizer(
            response,
            padding=False, return_tensors='pt', 
            add_special_tokens=False
        ).values()

        self.register_buffer(
            "instructEmbds",
            self.model.model.embed_tokens(instruct_ids.to(cfg.device)) # (L, D)
        )
        self.register_buffer(
            "responseEmbds",
            self.model.model.embed_tokens(response_ids.to(cfg.device)) # (L, D)
        )
        self.register_buffer(
            "instructMask", instruct_mask #(L, )
        )
        self.register_buffer(
            "responseMask", response_mask #(L, )
        )

        feats = freerec.utils.import_pickle(
            f"{cfg.dataset}_id_embeddings.pkl"
        )

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                feats,
                freeze=True,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.adaptor = nn.Linear(
            feats.size(1), self.model.config.hidden_size
        )
        self.output_proj = nn.Linear(
            self.model.config.hidden_size, self.Item.count + self.NUM_PADS,
            bias=False
        )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
        self.reset_parameters()

    def reset_parameters(self): ...
        # nn.init.xavier_normal_(self.adaptor.weight)
        # nn.init.xavier_normal_(self.output_proj.weight)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
           maxlen=maxlen, keep_at_least_itself=True
        ).seq_train_yielding_pos_(
            start_idx_for_target=-1, end_idx_for_input=-1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        B, L = seqs.shape

        instrutEmbds = self.instructEmbds.expand(B, -1, -1)
        instrutMask = self.instructMask.expand(B, -1)
        responseEmbds = self.responseEmbds.expand(B, -1, -1)
        responseMask = self.responseMask.expand(B, -1)

        seqEmbds = self.Item.embeddings(seqs)
        seqEmbds = self.adaptor(seqEmbds)

        inputs = torch.cat((instrutEmbds, seqEmbds, responseEmbds), dim=1)
        attnMask = torch.cat(
            (
                instrutMask,
                seqs.not_equal(self.PADDING_VALUE),
                responseMask
            ),
            dim=1
        )

        out = self.model(
            inputs_embeds=inputs,
            attention_mask=attnMask,
            return_dict=True
        )

        return out.last_hidden_state[:, -1]

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds = self.encode(data)
        logits = self.output_proj(userEmbds) # (B, N)
        labels = data[self.IPos].flatten()
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)
        return self.output_proj(userEmbds)[:, self.NUM_PADS:]

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)
        itemEmbds = self.output_proj.weight[self.NUM_PADS:]
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForE4SRec(freerec.launcher.Coach):

    def set_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay
        )

    def set_lr_scheduler(self):
        steps_per_epoch = math.ceil(sum([1 for _ in self.trainloader]) // cfg.gradient_accumulation_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=steps_per_epoch * cfg.epochs,
            num_warmup_steps=100
        )

    def train_per_epoch(self, epoch: int):
        step = 0
        scaler = torch.amp.GradScaler('cuda')
        for data in self.dataloader:
            step += 1
            data = self.dict_to_device(data)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = self.model(data)

            scaler.scale(loss).backward()
            if step % cfg.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                self.lr_scheduler.step()
           
            self.monitor(
                loss.item(), 
                n=data[self.Size], reduction="mean",
                mode='train', pool=['LOSS']
            )

        if step % cfg.gradient_accumulation_steps != 0:
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            self.lr_scheduler.step()
        
    def eval_at_best(self): ...


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.NextItemRecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = E4SRec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size)
    validpipe = model.sure_validpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size, ranking=cfg.ranking)

    coach = CoachForE4SRec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()