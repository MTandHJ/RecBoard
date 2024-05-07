

from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import coalesce
import freerec

from modules import EOPA, SGAT, AttnReadout

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)

cfg.set_defaults(
    description="LESSR",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class LESSR(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = cfg.embedding_dim,
        num_layers: int = cfg.num_layers,
        dropout_rate: float = cfg.dropout_rate,
        batch_norm: bool = True
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )

        input_dim = embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    activation=nn.PReLU(embedding_dim),
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
            minlen=2, keep_at_least_itself=True
        ).seq_train_yielding_pos_(
            start_idx_for_target=-1, end_idx_for_input=-1
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def get_multi_graphs(self, seqs: torch.Tensor, masks: torch.Tensor):
        def extract_from_seq(i: int, seq: np.ndarray, mask: np.ndarray):
            last = seq[-1]
            seq = seq[mask]
            items = np.unique(seq)
            mapper = {item:node for node, item in enumerate(items)}
            seq = [mapper[item] for item in seq]
            last = mapper[last]

            nums = len(items)
            x = torch.empty((nums, 0))
            
            # EOP
            graph_eop = Data(
                x=x,
                edge_index=torch.LongTensor(
                    [seq[:-1], seq[1:]]
                )
            )
            graph_eop.nodes = torch.from_numpy(items)

            # Shortcut
            graph_cut = Data(
                x=x,
                edge_index= coalesce(graph_eop.edge_index)
            )

            # Session
            graph_sess = Data(
                x=x,
                edge_index=torch.LongTensor([
                    [last] * nums, # for last items
                    [i] * nums,
                ])
            )

            return graph_eop, graph_cut, graph_sess
        graph_eop, graph_cut, graph_sess = zip(*map(extract_from_seq, range(len(seqs)), seqs.cpu().numpy(), masks.cpu().numpy()))
        # Batch graphs into a disconnected graph,
        # i.e., the edge_index will be re-ordered.
        graph_eop = Batch.from_data_list(graph_eop)
        graph_cut = Batch.from_data_list(graph_cut)
        graph_sess = Batch.from_data_list(graph_sess)
        return graph_eop.to(self.device), graph_cut.to(self.device), graph_sess.to(self.device)

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        masks = seqs.ne(0)
        graph_eop, graph_cut, graph_sess = self.get_multi_graphs(
            seqs, masks
        )

        features = self.Item.embeddings(graph_eop.nodes) # (*, D)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(features, graph_eop.edge_index)
            else:
                out = layer(features, graph_cut.edge_index)
            features = torch.cat([out, features], dim=1)

        last_features = features[graph_sess.edge_index[0]]
        groups = graph_sess.ptr
        groups = torch.repeat_interleave(
            torch.arange(len(groups) - 1, device=groups.device),
            groups[1:] - groups[:-1]
        )
        sr_g = self.readout(features, last_features, graph_sess.edge_index, groups)
        sr_l = features[graph_sess.edge_index[0].unique(sorted=False)]
        sr = torch.cat([sr_l, sr_g], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr)) # (B, D)
        return sr, self.Item.embeddings.weight[self.NUM_PADS:]

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode(data)

        logits = torch.einsum("BD,ND->BN", userEmbds, itemEmbds)
        targets = data[self.IPos].flatten() # (B,)
        rec_loss = self.criterion(logits, targets)
        return rec_loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForLESSR(freerec.launcher.Coach):

    def marked_params(self):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
        return params

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.marked_params(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )


    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = LESSR(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForLESSR(
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