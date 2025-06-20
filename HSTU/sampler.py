

from typing import Optional

import torchdata.datapipes as dp

import freerec
from freerec.data.datasets.base import RecDataSet
from freerec.data.tags import USER, ITEM, TIMESTAMP, ID, SEQUENCE
from freerec.data.postprocessing.source import RandomShuffledSource
from freerec.data.postprocessing.sampler import SeqTrainPositiveYielder, ValidSampler


def to_time_seqs(dataset: RecDataSet, maxlen: Optional[int] = None) -> RandomShuffledSource:
    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]
    Time = dataset.fields[TIMESTAMP]
    seqs = [[] for id_ in range(User.count)]
    timestamps = [[] for id_ in range(User.count)]

    dataset.listmap(
        lambda data: seqs[data[User]].append(data[Item]),
        dataset.to_rows(dataset[User, Item])
    )
    dataset.listmap(
        lambda data: timestamps[data[User]].append(data[Time]),
        dataset.to_rows(dataset[User, Time])
    )

    users = list(range(User.count))
    if maxlen is not None:
        seqs = [tuple(items[-maxlen:]) for items in seqs]
        timestamps = [tuple(times[-maxlen:]) for times in timestamps]
    else:
        seqs = [tuple(items) for items in seqs]
        timestamps = [tuple(times) for times in timestamps]

    source = dataset.to_rows(
        {User: users, Item.fork(SEQUENCE): seqs, Time.fork(SEQUENCE): timestamps}
    )
    return source

def shuffled_time_seqs_source(dataset: RecDataSet, maxlen: int) -> RandomShuffledSource:
    return RandomShuffledSource(dataset, to_time_seqs(dataset, maxlen))


@dp.functional_datapipe("time_seq_train_yielding_pos_")
class TimeSeqTrainPositiveYielder(SeqTrainPositiveYielder):

    def __init__(self, source, start_idx_for_target = 1, end_idx_for_input = -1):
        super().__init__(source, start_idx_for_target, end_idx_for_input)
        self.Time = self.fields[TIMESTAMP].fork(SEQUENCE)

    def __iter__(self):
        for row in self.source:
            seq = row[self.ISeq]
            timestamps = row[self.Time]
            if self._check(seq):
                row[self.IPos] = seq[self.start_idx_for_target:]
                row[self.ISeq] = seq[:self.end_idx_for_input]
                row[self.Time] = timestamps[:self.end_idx_for_input]
                yield row


@dp.functional_datapipe("time_valid_sampling_")
class TimeValidSampler(ValidSampler):

    def __init__(self, source, ranking = 'full', num_negatives = ...):
        super().__init__(source, ranking, num_negatives)

    @freerec.utils.timemeter
    def prepare(self):
        self.Time = self.fields[TIMESTAMP].fork(SEQUENCE)
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]
        seenTimestamps = [[] for _ in range(self.User.count)]
        unseenTimestamps = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs()
        )

        # Timestamps
        self.listmap(
            lambda row: seenTimestamps[row[self.User]].extend(row[self.Time]),
            to_time_seqs(self.dataset.train())
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs()
        )

        # Timestamps
        self.listmap(
            lambda row: unseenTimestamps[row[self.User]].extend(row[self.Time]),
            to_time_seqs(self.dataset.valid())
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.seenTimestamps = tuple(tuple(times) for times in seenTimestamps)
        self.unseenTimestamps = tuple(tuple(times) for times in unseenTimestamps)
        self.negItems = dict()

    def _nextitem_from_pool(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                timestamps = self.seenTimestamps[user] + self.unseenTimestamps[:k]
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                yield {self.User: user, self.ISeq: seq, self.Time: timestamps, self.IUnseen: unseen, self.ISeen: seen}

    def _nextitem_from_full(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                timestamps = self.seenTimestamps[user] + self.unseenTimestamps[:k]
                unseen = (positive,)
                yield {self.User: user, self.ISeq: seq, self.Time: timestamps, self.IUnseen: unseen, self.ISeen: seen}


@dp.functional_datapipe("time_test_sampling_")
class TimeTestSampler(TimeValidSampler):

    @freerec.utils.timemeter
    def prepare(self):
        self.Time = self.fields[TIMESTAMP].fork(SEQUENCE)
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]
        seenTimestamps = [[] for _ in range(self.User.count)]
        unseenTimestamps = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs()
        )

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs()
        )

        # Timestamps
        self.listmap(
            lambda row: seenTimestamps[row[self.User]].extend(row[self.Time]),
            to_time_seqs(self.dataset.train())
        )

        # Timestamps
        self.listmap(
            lambda row: seenTimestamps[row[self.User]].extend(row[self.Time]),
            to_time_seqs(self.dataset.valid())
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.test().to_seqs()
        )

        # Timestamps
        self.listmap(
            lambda row: unseenTimestamps[row[self.User]].extend(row[self.Time]),
            to_time_seqs(self.dataset.test())
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.seenTimestamps = tuple(tuple(times) for times in seenTimestamps)
        self.unseenTimestamps = tuple(tuple(times) for times in unseenTimestamps)
        self.negItems = dict()