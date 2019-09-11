# This piece of code has been developed by Miquel Esplà-Gomis [mespla@dlsi.ua.es]
# and is distributed under GPL v3 [https://www.gnu.org/licenses/gpl-3.0.html]
# (c) 2019 Universitat d'Alacant (http://www.ua.es)

# Code based on Fairseq [https://github.com/pytorch/fairseq/]

import math

from fairseq import utils
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion


@register_criterion('cross_entropy_with_langid')
class CrossEntropyCriterionWithLangID(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, lang, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(lang, **sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
