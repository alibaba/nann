from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from typing import List, Optional
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
        '_normal_layer',
        'BatchNormCaffe',
        'PlaceHolderLayer',
        'layer_weights_init',
        'BlurPooling',
        'CARAFE',
        'DistributedBatchNorm1d',
        'DistributedBatchNorm2d',
        'DistributedBatchNorm3d',
        ]

def _normal_layer(bn_type, *args, **kwargs):
    if bn_type == 'caffe':
        return BatchNormCaffe(*args, **kwargs)
    elif bn_type == 'group':
        return nn.GroupNorm(*args, **kwargs)
    elif bn_type == 'spectral':
        return PlaceHolderLayer()
    elif bn_type == 'dist_batch':
        return DistributedBatchNorm2d(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)

class BatchNormCaffe(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 use_global_status=True):
        super(BatchNormCaffe, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.use_global_status = use_global_status
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight.data.uniform_()
            self.bias.data.zero_()
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        if self.affine:
            if self.training and not self.use_global_status:
                mean = input.mean(dim=0, keepdim=True)\
                    .mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                var = (input-mean)*(input-mean).mean(dim=0, keepdim=True).\
                    mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

                self.running_mean = (mean.squeeze()
                                     + (1-self.momentum)*self.running_mean)/(2-self.momentum)
                batch = input.size()[0]
                bias_correction_factor = batch-1 if batch>1 else 1
                self.running_var = (var.squeeze() * bias_correction_factor
                                    + (1-self.momentum)*self.running_var)/(2-self.momentum)
                x = input - mean
                x = x / (var+self.eps).sqrt()

                tmp_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = x * tmp_weight
                x = x + tmp_bias
            else:
                tmp_running_mean = self.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_running_var = self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = input - tmp_running_mean
                x = x / (tmp_running_var+self.eps).sqrt()

                tmp_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = x * tmp_weight
                x = x + tmp_bias
        else:
            x = input
        return x

    def extra_repr(self):
        s = ('{num_features}, eps={eps}, momentum={momentum}'
             ', affine={affine}, use_global_status={use_global_status}')
        return s.format(**self.__dict__)

class PlaceHolderLayer(nn.Module):
    def forward(self, input):
        return input

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    # weight[range(in_channels), range(out_channels), :, :] = filt
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return torch.from_numpy(weight)

def layer_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        # init.normal(m.weight.data, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                length = len(param)
                nn.init.constant(param, 0.0)
                nn.init.constant(param[length // 4:length // 2], 1.0)
            elif 'weight' in name:
                nn.init.uniform(param, -0.2, 0.2)
                # nn.init.xavier_normal(param)
    elif isinstance(m, nn.ConvTranspose2d):
        size = m.weight.data.size()
        m.weight.data = bilinear_kernel(size[0], size[1], size[2])
        if m.bias is not None:
            init.constant_(m.bias, 0)

class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP]. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has ``decode`` method which finds the
    best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Arguments
    ---------
    num_tags : int
        Number of tags.
    batch_first : bool, optional
        Whether the first dimension corresponds to the size of a minibatch.
    Attributes
    ----------
    start_transitions : :class:`~torch.nn.Parameter`
        Start transition score tensor of size ``(num_tags,)``.
    end_transitions : :class:`~torch.nn.Parameter`
        End transition score tensor of size ``(num_tags,)``.
    transitions : :class:`~torch.nn.Parameter`
        Transition score tensor of size ``(num_tags, num_tags)``.
    References
    ----------
    .. [LMP] Lafferty, J., McCallum, A., Pereira, F. (2001).
             "Conditional random fields: Probabilistic models for segmenting and
             labeling sequence data". *Proc. 18th International Conf. on Machine
             Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags, batch_first=False):
        if num_tags <= 0:
            raise ValueError('invalid number of tags: {}'.format(num_tags))
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self):
        return '{}(num_tags={})'.format(self.__class__.__name__, self.num_tags)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Arguments
        ---------
        emissions : :class:`~torch.Tensor`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length, num_tags)`` otherwise.
        tags : :class:`~torch.LongTensor`
            Sequence of tags tensor of size ``(seq_length, batch_size)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        mask : :class:`~torch.ByteTensor`, optional
            Mask tensor of size ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length)`` otherwise.
        reduction : str, optional
            Specifies  the reduction to apply to the output: 'none'|'sum'|'mean'|'token_mean'.
            'none': no reduction will be applied. 'sum': the output will be summed over batches.
            'mean': the output will be averaged over batches. 'token_mean': the output will be
            averaged over tokens.
        Returns
        -------
        :class:`~torch.Tensor`
            The log likelihood. This will have size ``(batch_size,)`` if reduction is 'none',
            ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError('invalid reduction: {}'.format(reduction))
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Arguments
        ---------
        emissions : :class:`~torch.Tensor`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length, num_tags)`` otherwise.
        mask : :class:`~torch.ByteTensor`, optional
            Mask tensor of size ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length)`` otherwise.
        Returns
        -------
        List[List[int]]
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError('emissions must have dimension of 3, got {}'.format(emissions.dim()))
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                'expected last dimension of emissions is {}, '
                'got {}'.format(self.num_tags, emissions.size(2)))

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    'got {} and {}'.format(tuple(emissions.shape[:2]), tuple(tags.shape)))

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    'got {} and {}'.format(tuple(emissions.shape[:2]), tuple(mask.shape)))
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

'''
    Antialiased CNNs
    https://github.com/adobe/antialiased-cnns
'''
class BlurPooling(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(BlurPooling, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    def get_pad_layer(self, pad_type):
        if(pad_type in ['refl','reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl','replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized'%pad_type)
        return PadLayer

'''
    CARAFE: Content-Aware ReAssembly of FEatures
'''
class CARAFE(nn.Module):
    def __init__(self, in_ch, in_h_w, scale,
                 kernel_size_en=3, kernel_size_up=5,
                 ch_compression=64):
        super(CARAFE, self).__init__()
        self.in_h_w = in_h_w
        self.scale = scale
        self.compress_conv = nn.Conv2d(
            in_ch, ch_compression, kernel_size=1,
        )
        self.kernel_conv = nn.Conv2d(
            ch_compression, kernel_size_up**2*(in_h_w*scale)**2,
            kernel_size=kernel_size_en,
            padding=(kernel_size_en - 1) // 2,
        )

    def forward(self, input):
        shape = input.shape
        up_kernel = self.compress_conv(input)
        up_kernel = self.kernel_conv(up_kernel)

        # shape: N, scale**2 * up_kernel**2, H, W -> N, 1, scale**2, up_kernel**2, H, W
        up_kernel = up_kernel.reshape(shape[0], self.scale**2, -1, shape[2], shape[3])
        up_kernel = F.softmax(up_kernel, 2)
        up_kernel = up_kernel.unsqueeze(1)

        # shape: N, C, H, W -> N, C, 1, 1, H, W
        input = input.unsqueeze(2).unsqueeze(2)
        output = up_kernel * input

        # sum
        output = output.sum(3)

        # shape: N, C, scale**2, up_kernel**2, H, W -> N, C, scale*H, scale*W
        output = output.reshape(shape[0], shape[1], self.scale*shape[2], self.scale*shape[3])
        return output

'''
    DistributedBatchNorm
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
'''
def all_reduce(tensors):
    async_op=False
    requests = []
    for p in tensors:
        r = dist.all_reduce(p)
        requests.append(r)
    if async_op:
        for r in requests:
            r.wait()

def all_reduce_async(tensors):
    requests = []
    for p in tensors:
        r = dist.all_reduce(p, async_op=True)
        requests.append(r)
    return requests

def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)

def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)

class _DistributedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 grad_mode='exact', is_async=False):
        super(_DistributedBatchNorm, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine)

        self.grad_mode = grad_mode
        self._async = is_async
        self.distributed = True
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            self.distributed = False

        if affine:
            init.zeros_(self.bias)
            init.ones_(self.weight)

        if self._async:
            self._requests = []
            self._bufs = [torch.zeros(num_features).cuda(),
                          torch.ones(num_features).cuda()]
            self._warmup = False

    def forward(self, input):
        # if not (self.distributed and self.training):
        #     return F.batch_norm(
        #         input, self.running_mean, self.running_var, self.weight, self.bias,
        #         self.training, self.momentum, self.eps)

        if self.distributed and self.training:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        size = sum_size * self.world_size
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        if not (self.distributed and self.training):
            mean, inv_std = self._compute_mean_std(input_sum, input_ssum, size)
        elif self._async:
            self._sync()
            if self._warmup:
                mean, inv_std = self._compute_mean_std(self._bufs[0], self._bufs[1], size)
            else:
                mean, inv_std = self._compute_mean_std(input_sum, input_ssum, size)
                self._warmup = True
            self._bufs[0].copy_(input_sum.data)
            self._bufs[1].copy_(input_ssum.data)
            self._requests = all_reduce_async(self._bufs)
        else:
            _input_sum = input_sum.expand_as(input_sum)
            _input_ssum = input_ssum.expand_as(input_ssum)
            all_reduce([_input_sum, _input_ssum])
            if self.grad_mode == 'approx':
                _input_sum += (self.world_size - 1) * (input_sum - input_sum.data)
                _input_ssum += (self.world_size - 1) * (input_ssum - input_ssum.data)
            else:
                def _reduce_grad(grad):
                    _grad = grad + 0
                    all_reduce([_grad])
                    return _grad

                input_sum.register_hook(_reduce_grad)
                input_ssum.register_hook(_reduce_grad)
            mean, inv_std = self._compute_mean_std(_input_sum, _input_ssum, size)

        mean = _unsqueeze_ft(mean)
        if self.affine:
            output = (input - mean) * _unsqueeze_ft(inv_std * self.weight) \
                     + _unsqueeze_ft(self.bias)
        else:
            output = (input - mean) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def _sync(self):
        for r in self._requests:
            r.wait()

    def _compute_mean_std(self, input_sum, input_ssum, size):
        if not self.training:
            return self.running_mean, (self.running_var + self.eps) ** -0.5

        mean = input_sum / size
        sumvar = input_ssum - input_sum * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
        if self._async:
            return self.running_mean, self.running_var.clamp(min=self.eps) ** -0.5
        else:
            return mean, (bias_var + self.eps) ** -0.5

class DistributedBatchNorm1d(_DistributedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class DistributedBatchNorm2d(_DistributedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class DistributedBatchNorm3d(_DistributedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

if __name__ == '__main__':
    layer = BatchNormCaffe(3, use_global_status=False)
    layer.train()

    # print(layer.running_mean.size())
    input = torch.ones(1, 3, 3, 3)
    output = layer(input)
    print(output, layer.running_mean.size())
