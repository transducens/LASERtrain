# This piece of code has been developed by Miquel Esplà-Gomis [mespla@dlsi.ua.es]
# and is distributed under GPL v3 [https://www.gnu.org/licenses/gpl-3.0.html]
# (c) 2019 Universitat d'Alacant (http://www.ua.es)

# Code based on Fairseq [https://github.com/pytorch/fairseq/] and LASER [https://github.com/facebookresearch/LASER]

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
    FairseqIncrementalDecoder,
)
from fairseq.models.lstm import (
    base_architecture,
    Embedding,
    LSTMModel,
    LSTMEncoder,
    LSTMDecoder,
    LSTMCell,
    Linear,
)

from . import multilingual_translation_single_model

@register_model('multilingual_lstm_laser')
class MultilingualLSTMModelLaser(FairseqEncoderDecoderModel):
    """Train LSTM models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from LSTMModel and assume that all language
    pairs use a single LSTM architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.shared_dict=None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        LSTMModel.add_args(parser)
        parser.add_argument('--share-dictionaries', action='store_true',
                            help='share word dictionaries across languages')
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--lang-embedding-size', type=int, default=32,
                            help='size of the language embedding')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert isinstance(task, multilingual_translation_single_model.MultilingualTranslationTaskWithSingleModel)

        #Piece of code to make sure that all dictionaries are the same
        shared_dict = task.dicts[task.langs[0]]
        if any(task.dicts[lang] != shared_dict for lang in task.langs):
            raise ValueError('This model only uses shared dictionaries: all the dictionaries provided must have the same size.')

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        # Languages index: lang codes into integers
        lang_dictionary = { task.langs[i] : i for i in range(0, len(task.langs)) }

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings for encoder and decoder, as well as embedding for language cores
        shared_embed_tokens = build_embedding(shared_dict, args.encoder_embed_dim)
        shared_language_embeddings = nn.Embedding(len(lang_dictionary), args.lang_embedding_size)

        # Building encoder and decoder; Encoder has been copied and adapted from LASER package [https://github.com/facebookresearch/LASER]
        shared_encoder=LaserEncoder(shared_dict, shared_embed_tokens, embed_dim=args.encoder_embed_dim, num_layers=args.encoder_layers, bidirectional=args.encoder_bidirectional)
        shared_decoder=MultilangLSTMDecoder(shared_dict, lang_dictionary, shared_embed_tokens, shared_language_embeddings, args.decoder_embed_dim, encoder_output_units=int(args.encoder_hidden_size)*2, hidden_size=args.decoder_hidden_size, out_embed_dim=args.decoder_embed_dim, attention=False, share_input_output_embed=False, lang_embedding_size=args.lang_embedding_size)

        def get_encoder(lang):
            return shared_encoder

        def get_decoder(lang):
            return shared_decoder

        return MultilingualLSTMModelLaser(shared_encoder, shared_decoder)

    def load_state_dict(self, state_dict, strict=True):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('model.')
        super().load_state_dict(state_dict_subset, strict=strict)

    def forward(self, decoder_lang, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, lang=decoder_lang, **kwargs)
        return decoder_out

    def __call__(self, lang, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(lang, *input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result




class MultilangLSTMDecoder(FairseqIncrementalDecoder):
    def __init__(
        self, dictionary, lang_dictionary, embedding, lang_embedding, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=1024, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None, lang_embedding_size=32
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.lang_embeddings_size = lang_embedding_size
        self.lang_dictionary = lang_dictionary
        self.embed_langs = lang_embedding

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = embedding

        self.encoder_output_units = encoder_output_units

        self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
        self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim + self.lang_embeddings_size + self.encoder_output_units if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    #Forward receives language
    def forward(self, prev_output_tokens, encoder_out, lang, incremental_state=None):
        encoder_sentemb = encoder_out['sentemb']
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # embed language
        lang_tensor=torch.cuda.LongTensor([self.lang_dictionary[lang]]*bsz)
        l = self.embed_langs(lang_tensor)

        # B x T x C -> T x B x C
        #l = l.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
            print(len(prev_cells[0]))
        else:
            num_layers = len(self.layers)
            # Hiddens and cells are initialized with a linear transformation of the embedding produced by the encoder 
            prev_hiddens = [encoder_sentemb for i in range(num_layers)]
            prev_cells = [encoder_sentemb for i in range(num_layers)]
            prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
            prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], encoder_sentemb, input_feed, l), dim=1)


            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x, None

################################################
########### CODE FROM LASER PROJECT ############
################################################

def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


# TODO Do proper padding from the beginning
def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)

#Need to make it inherit from FairseqEncoder to be able to use the FaiseqMulti model
class LaserEncoder(FairseqEncoder):
    def __init__(
        self, dictionary, embedding, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=True,
        left_pad=True, padding_value=0.
    ):
        super().__init__(dictionary)

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = dictionary.pad()
        self.embed_tokens = embedding

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb': sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

@register_model_architecture('multilingual_lstm_laser', 'multilingual_lstm_laser')
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', True)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', True)
    args.share_encoders = getattr(args, 'share_encoders', True)
    args.share_decoders = getattr(args, 'share_decoders', True)

@register_model_architecture('multilingual_lstm_laser', 'multilingual_lstm_laser_mseLearn')
def multilingual_lstm_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 2048)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    base_multilingual_architecture(args)
