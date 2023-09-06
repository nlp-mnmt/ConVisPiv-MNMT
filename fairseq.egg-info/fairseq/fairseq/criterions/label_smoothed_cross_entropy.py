# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
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

    def compute_loss(self, model, net_output, sample, reduce=True):
        net_output_x = net_output[0:2]
        net_output_x_txt = net_output[2:4]
        net_output_x_img = net_output[4:6]
        # net_output_x_img_no = net_output[6:8]


        lprobs_txt = model.get_normalized_probs(net_output_x, log_probs=True)
        lprobs_x_txt = model.get_normalized_probs(net_output_x_txt, log_probs=True)
        lprobs_x_img = model.get_normalized_probs(net_output_x_img, log_probs=False)
        # lprobs_x_img_no = model.get_normalized_probs(net_output_x_img_no, log_probs=False)

        lprobs = lprobs_txt.view(-1, lprobs_txt.size(-1))
        # lprobs_img = lprobs_img.view(-1, lprobs.size(-1))

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)

        target = target.view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # txt_out = net_output[0]
        # img_out = net_output[2]

        lprobs_x_txt = torch.exp(lprobs_x_txt)
        lprobs_x_img = torch.exp(lprobs_x_img)
        # lprobs_x_img_no = torch.exp(lprobs_x_img_no)
        lprobs_x_imgs_no = torch.zeros(lprobs_x_img)
        consis_loss_help = utils.multimodel_consis_loss(lprobs_x_txt, lprobs_x_img)
        consis_loss_help_no = utils.multimodel_consis_loss(lprobs_x_txt, lprobs_x_imgs_no)
        # if consis_loss_help > 0:
        #     if consis_loss_help_no > 0:
        #         consis_loss_all = consis_loss_help + consis_loss_help_no
        #     else:
        #         consis_loss_all = consis_loss_help - consis_loss_help_no
        # if consis_loss_help <= 0:
        #     if consis_loss_help_no <= 0:
        #         consis_loss_all = -consis_loss_help - consis_loss_help_no
        #     else:
        #         consis_loss_all = -consis_loss_help + consis_loss_help_no
        # consis_loss_all = -math.fabs(consis_loss_help) + math.fabs(consis_loss_help_no)
        # consis_loss_all = torch.einsum('nc,nc->n',[lprobs_x_txt,lprobs_x_img])
        # consis_loss_all = torch.log(-consis_loss_help_no)
        consis_loss_all = -torch.log(consis_loss_help / consis_loss_help_no)

        # alpha = 0.8
        # consis_loss_all = alpha * consis_loss_all
        # consis_loss_all = math.fabs(consis_loss_help_no)

        # print(consis_loss_all)

        return loss + consis_loss_all , nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
