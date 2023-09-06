# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import sys
import types
import torch
import numpy as np


logger = logging.getLogger(__name__)


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate_multimodel_graphs(txt_graphs, img_txt_graphs, src_lengths):
    max_txt_len = max(src_lengths).item()
    max_img_len = max([i.shape[0] for i in img_txt_graphs])
    assert len(txt_graphs) == len(img_txt_graphs), 'txt lengths and image2txt lengths must be same.'
    multimodel_gragh = torch.zeros(len(txt_graphs), max_img_len + max_txt_len, max_txt_len)

    def copy_and_merge_graphs(per_src_tensor, per_txt_graph, per_img_txt_graph):
        per_src_tensor[:per_txt_graph.shape[0],:per_txt_graph.shape[0]] = torch.tensor(per_txt_graph).type_as(per_src_tensor)
        per_src_tensor[len(per_src_tensor) - per_img_txt_graph.shape[0]:,
                        :per_img_txt_graph.shape[1]] = torch.tensor(per_img_txt_graph).type_as(per_src_tensor)

    for i, per_item in enumerate(zip(multimodel_gragh, txt_graphs, img_txt_graphs)):
        copy_and_merge_graphs(per_item[0], per_item[1], per_item[2])
    return multimodel_gragh






# def collate_graph_tokens(values, src_len, pad_idx):
#     """Convert a list of 1d tensors into a padded 2d tensor."""
#     # import pdb
#     # pdb.set_trace()
#     # print(values)
#     size = max(v.shape[0] for v in values)
#     # res = values[0].new(len(values), size, size).fill_(pad_idx)
#     res = pad_idx*np.ones([len(values), size, size])

def collate_graph_tokens(values, src_len_tensor):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(src_len_tensor).item()
    # print('size',size)
    res = torch.zeros([len(values), size, size])

    def copy_tensor(src, dst):
        # print('src', src.size())
        # print('dst', dst.size())
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        tmp_src_len = len(res[i])
        # print('tmp_src_len', tmp_src_len)
        v = torch.tensor(v)
        # print('v', len(v))
        # if tmp_src_len == size and len(v) >= tmp_src_len:
        #     print('if')
        #     copy_tensor(v[0:v.shape[0], 0:v.shape[0]], res[i])
        # else:
        #     print('else')
        #     copy_tensor(v, res[i][1:len(v) + 1, 1:len(v) + 1])
        if len(v) > tmp_src_len:
            try:
                copy_tensor(v[1:len(res[i])+1, 1:len(res[i])+1], res[i])
            except:
                123
        elif len(v) == tmp_src_len:
            try:
                copy_tensor(v, res[i])
            except:
                123
            # print('elif')
        else:
            # print('else')
            try:
                copy_tensor(v, res[i][1:len(v) + 1, 1:len(v) + 1])
            except:
                123
    return res


def load_indexed_dataset(path, dictionary, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        logger.info('loaded {} examples from: {}'.format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # Hacky as heck, for the specific case of multilingual training with RoundRobin.
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):
                return all(
                    a is None or b is None or a <= b
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )
    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception((
            'Size of sample #{} is invalid (={}) since max_positions={}, '
            'skip this example with --skip-invalid-size-inputs-valid-test'
        ).format(ignored[0], dataset.size(ignored[0]), max_positions))
    if len(ignored) > 0:
        logger.warn((
            '{} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))
    return indices


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    try:
        from fairseq.data.data_utils_fast import batch_by_size_fast
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    return batch_by_size_fast(indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult)


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol == '_EOW':
        sentence = sentence.replace(' ', '').replace('_EOW', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence

def load_sp_graph(graph_npy_path):
    graph_npy = np.load(graph_npy_path + '.npy', allow_pickle=True)
    graph_tensor = [np.array(lin.todense()) for lin in graph_npy]
    return graph_tensor

def load_img_features(imag_npy_path):
    img_npy = np.load(imag_npy_path + '.npy', allow_pickle=True)
    img_tensor = torch.tensor(img_npy)
    img_tensor = img_tensor.view(img_tensor.size(0),img_tensor.size(1)*img_tensor.size(2),-1)
    return img_tensor

def load_multimodel_graph(multimodel_graph_path):
    multimodel_graph = np.load(multimodel_graph_path + 'npy', allow_pickle=True)
    bpe_relations = [itm['bpe_relation'] for itm in multimodel_graph]
    img_txt_relations = [itm['img_txt_relation'] for itm in multimodel_graph]
    return bpe_relations, img_txt_relations



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
