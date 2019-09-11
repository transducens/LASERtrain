# This piece of code has been developed by Miquel Esplà-Gomis [mespla@dlsi.ua.es]
# and is distributed under GPL v3 [https://www.gnu.org/licenses/gpl-3.0.html]
# (c) 2019 Universitat d'Alacant (http://www.ua.es)

# Code based on Fairseq [https://github.com/pytorch/fairseq/]

from collections import OrderedDict
import copy
import os

import torch

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    data_utils,
    iterators,
    LanguagePairDataset,
    FairseqDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    indexed_dataset,
)
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.tasks.translation import load_langpair_dataset 
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


from fairseq.tasks import FairseqTask, register_task


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


@register_task('multilingual_translation_singlemodel')
class MultilingualTranslationTaskWithSingleModel(MultilingualTranslationTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

    def build_model(self, args):
        def check_args():
            messages = []
            if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
                messages.append('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append('--encoder-langtok should be {}.'.format(args.encoder_langtok))
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append('--decoder-langtok should {} be set.'.format("" if args.decoder_langtok else "not"))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqEncoderDecoderModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
        return model


    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        for lang_pair in self.args.lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model, sample[lang_pair], lang_pair.split("-")[1])
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model, sample[lang_pair], lang_pair.split("-")[1])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output


    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)


    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        def dataset_size(index):
            s = dataset.size(index)
            return list(s.values())[0]

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset_size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

