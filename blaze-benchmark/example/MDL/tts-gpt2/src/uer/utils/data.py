# -*- encoding:utf-8 -*-
import os
import torch
import codecs
import random
import pickle
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed

from PIL import Image
import cv2
import torchvision.transforms as transforms

from torch.utils import data
import linecache
import json



def mask_seq(src, vocab_size):
    """
    mask input sequence for MLM task
    args:
        src: a list of tokens
        vocab_size: the vocabulary size
    """
    tgt_mlm = []
    for (i, token) in enumerate(src):
        if token == CLS_ID or token == SEP_ID:
            continue
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                src[i] = MASK_ID
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, vocab_size-1)
                    if rdi not in [CLS_ID, SEP_ID, MASK_ID]:
                        break
                src[i] = rdi
            tgt_mlm.append((i, token))
    return src, tgt_mlm


def merge_dataset(dataset_path, workers_num):
        # Merge datasets.
        f_writer = open(dataset_path, "wb")
        for i in range(workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                tmp_data = tmp_dataset_reader.read(2^20)
                if tmp_data:
                    f_writer.write(tmp_data)
                else:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed

    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert(workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.f_read = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.f_read)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.f_read.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.f_read.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """
    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1
                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, f_write)
                        # Clear buffer.
                        docs_buffer = []
                        instances = []
                    continue
                sentence = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(sentence) > 0:
                    document.append(sentence)

                if pos >= end - 1:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, f_write)
                    break
        f_write.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_random_next = 0

                    # Random next
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual next
                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    # assert len(tokens_a) >= 1
                    # assert len(tokens_b) >= 1

                    src = []

                    src.append(CLS_ID)
                    for token in tokens_a:
                        src.append(token)

                    src.append(SEP_ID)

                    seg_pos = [len(src)]

                    for token in tokens_b:
                        src.append(token)

                    src.append(SEP_ID)

                    seg_pos.append(len(src))

                    src, tgt_mlm = mask_seq(src, len(self.vocab))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    instance = (src, tgt_mlm, is_random_next, seg_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """ truncate sequence pair to specific length """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            # assert len(trunc_tokens) >= 1

            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue

            for ins in instances:
                src.append(ins[0])
                tgt_mlm.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt_mlm[-1][mask[0]] = mask[1]
                is_next.append(ins[2])
                seg.append([1]*ins[3][0] + [2]*(ins[3][1]-ins[3][0]) + [PAD_ID]*(len(ins[0])-ins[3][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [SOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(line)] + [EOS_ID]
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class ClmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 2:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [EOS_ID]
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class ClmpropDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[2])] + [EOS_ID]
                if len(target) > self.seq_length / 2:
                    continue
                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])]
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]
                src = source + target
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class Seq2seqDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[2])] + [EOS_ID]
                if len(target) > self.seq_length / 2:
                    continue
                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])]
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class FpdgDataset(Dataset):
    def set_type_vocab(self, type_vocab):
        self.type_vocab = type_vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                target_type = [SOS_ID]
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    target += word_tokens
                    target_type += [self.type_vocab.get(pair_list[1].strip())] * len(word_tokens)
                target += [EOS_ID]
                target_type += [EOS_ID]
                if len(target) > self.seq_length / 2:
                    continue

                source = []
                source_type = []
                for pair in tmp_list[0].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    source += word_tokens
                    source_type += [self.type_vocab.get(pair_list[1].strip())] * len(word_tokens)
                source += [DOS_ID]
                source_type += [DOS_ID]
                '''
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    source += word_tokens
                    source_type += [self.type_vocab.get(pair_list[1].strip())] * len(word_tokens)
                '''
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1] + ":" + pair_list[0])]
                    source += word_tokens
                    # hard code item prop ner type
                    source_type += [5] * len(word_tokens)
                source = source[:int(self.seq_length / 2 - 1)]
                source_type = source_type[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]
                source_type += [DOS_ID]

                src_len = len(source)
                tgt_len = len(target)

                src = source + target[:-1]
                src_type = source_type + target_type[:-1]
                tgt = [0] * src_len + target[1:]
                tgt_type = [0] * src_len + target_type[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    src_type = src_type[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    tgt_type = tgt_type[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        src_type.append(PAD_ID)
                        tgt.append(PAD_ID)
                        tgt_type.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, src_type, tgt, tgt_type, seg, src_len, tgt_len), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class VaeDataset(Dataset):
    def set_condition_length(self, condition_length):
        self.condition_length = condition_length

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 4:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                for pair in tmp_list[3].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    target += word_tokens
                target += [EOS_ID]

                src_len = 0
                tgt_len = len(target)

                src = target[:-1]
                tgt = target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                prop = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1] + ":" + pair_list[0])]
                    prop += word_tokens

                condition_title = []
                word_list = tmp_list[0].split("\001")
                random.shuffle(word_list)
                for pair in word_list:
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    condition_title += word_tokens
                condition_title += prop
                condition_title_seg = [1] * len(condition_title)
                condition_title_length = self.condition_length * 2
                if len(condition_title) >= condition_title_length:
                    condition_title = condition_title[:condition_title_length]
                    condition_title_seg = condition_title_seg[:condition_title_length]
                else:
                    while len(condition_title) != condition_title_length:
                        condition_title.append(PAD_ID)
                        condition_title_seg.append(PAD_ID)

                condition_text = []
                word_list = tmp_list[1].split("\001")
                random.shuffle(word_list)
                for pair in word_list:
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    condition_text += word_tokens
                condition_text_seg = [1] * len(condition_text)
                condition_text_length = self.condition_length
                if len(condition_text) >= condition_text_length:
                    condition_text = condition_text[:condition_text_length]
                    condition_text_seg = condition_text_seg[:condition_text_length]
                else:
                    while len(condition_text) != condition_text_length:
                        condition_text.append(PAD_ID)
                        condition_text_seg.append(PAD_ID)

                pickle.dump((condition_title, condition_text, src, tgt, seg, src_len, tgt_len, condition_title_seg, condition_text_seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StorylineDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                word_list = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    word_list.append(pair_list[0])
                target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
                target += [EOS_ID]
                if len(target) > self.seq_length / 2:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID]
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    word_tokens = [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1] + ":" + pair_list[0])]
                    source += word_tokens
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StorylinepropDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                word_list = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    word_list.append(pair_list[0])
                target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
                target += [EOS_ID]
                if len(target) > self.seq_length / 4 * 3:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                source = source[:int(self.seq_length / 4 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len, prop_keys, prop_values), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StorylinepropattrDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        import json
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 4:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                word_list = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    word_list.append(pair_list[0])
                target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
                target += [EOS_ID]
                if len(target) > self.seq_length / 4 * 3:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])] + [DOS_ID]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                try:
                    ATTR_LEN = 10
                    ATTR_NUM = 50
                    attr_keys = []
                    attr_values = []
                    json_obj = json.loads(tmp_list[3])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        key = pair['property_name']
                        value = pair['values'][0]['value_name']
                        attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]
                        attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                        if len(attr_key) >= ATTR_LEN:
                            attr_key = attr_key[:ATTR_LEN]
                        else:
                            while len(attr_key) != ATTR_LEN:
                                attr_key.append(PAD_ID)
                        if len(attr_value) >= ATTR_LEN:
                            attr_value = attr_value[:ATTR_LEN]
                        else:
                            while len(attr_value) != ATTR_LEN:
                                attr_value.append(PAD_ID)

                        attr_keys.append(attr_key)
                        attr_values.append(attr_value)

                    if len(attr_keys) >= ATTR_NUM:
                        attr_keys = attr_keys[:ATTR_NUM]
                        attr_values = attr_values[:ATTR_NUM]
                    else:
                        while len(attr_keys) != ATTR_NUM:
                            attr_keys.append([PAD_ID] * ATTR_LEN)
                            attr_values.append([PAD_ID] * ATTR_LEN)
                except:
                    continue

                source = source[:int(self.seq_length / 4 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len, prop_keys, prop_values, attr_keys, attr_values), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class ShorthighlightpropattrDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        import json
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 5:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                target += [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[3])]
                target += [EOS_ID]
                if len(target) > self.seq_length / 2:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                try:
                    ATTR_LEN = 10
                    ATTR_NUM = 50
                    attr_keys = []
                    attr_values = []
                    json_obj = json.loads(tmp_list[4])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        key = pair['property_name']
                        value = pair['values'][0]['value_name']
                        attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]
                        attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                        if len(attr_key) >= ATTR_LEN:
                            attr_key = attr_key[:ATTR_LEN]
                        else:
                            while len(attr_key) != ATTR_LEN:
                                attr_key.append(PAD_ID)
                        if len(attr_value) >= ATTR_LEN:
                            attr_value = attr_value[:ATTR_LEN]
                        else:
                            while len(attr_value) != ATTR_LEN:
                                attr_value.append(PAD_ID)

                        attr_keys.append(attr_key)
                        attr_values.append(attr_value)

                    if len(attr_keys) >= ATTR_NUM:
                        attr_keys = attr_keys[:ATTR_NUM]
                        attr_values = attr_values[:ATTR_NUM]
                    else:
                        while len(attr_keys) != ATTR_NUM:
                            attr_keys.append([PAD_ID] * ATTR_LEN)
                            attr_values.append([PAD_ID] * ATTR_LEN)
                except:
                    continue

                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len, prop_keys, prop_values, attr_keys, attr_values), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class Shorthighlightseq2seqDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 5:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[3])] + [EOS_ID]
                if len(target) > self.seq_length / 4:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    source += [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]

                try:
                    json_obj = json.loads(tmp_list[4])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        value = pair['values'][0]['value_name']
                        source += [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]
                except:
                    print('json format error!')
                    continue

                source = source[:int(self.seq_length / 4 * 3 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg, src_len, tgt_len), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class ShorthighlightclmpropattrDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 5:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[3])] + [EOS_ID]
                if len(target) > self.seq_length / 4:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

                prop_values = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    prop_values.append(pair_list[0])
                source += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(prop_values))]
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]

                try:
                    attr_values = []
                    json_obj = json.loads(tmp_list[4])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        value = pair['values'][0]['value_name']
                        attr_values.append(value)
                    source += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(attr_values))]
                except:
                    print('json format error!')
                    continue

                source = source[:int(self.seq_length / 4 * 3 - 1)]
                source += [DOS_ID]
                src = source + target
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class ShorthighlightclmpropattrkeywordDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 6:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[3])] + [EOS_ID]
                if len(target) > self.seq_length / 4:
                    continue

                if not tmp_list[5].strip():
                    continue
                keywords_list = tmp_list[5].split("\001")
                random.shuffle(keywords_list)
                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(keywords_list[0].split("\002")[0])] + [DOS_ID]

                source += [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

                prop_values = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    prop_values.append(pair_list[0])
                source += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(prop_values))]
                source = source[:int(self.seq_length / 2 - 1)]
                source += [DOS_ID]

                try:
                    attr_values = []
                    json_obj = json.loads(tmp_list[4])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        value = pair['values'][0]['value_name']
                        attr_values.append(value)
                    source += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(attr_values))]
                except:
                    print('json format error!')
                    continue

                source = source[:int(self.seq_length / 4 * 3 - 1)]
                source += [DOS_ID]
                src = source + target
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StorylinepropattrpictDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        Dataset.__init__(self, args, vocab, tokenizer)

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        import json
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 5:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                target = [SOS_ID]
                word_list = []
                for pair in tmp_list[3].split("\001"):
                    pair_list = pair.split("\002")
                    word_list.append(pair_list[0])
                target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
                target += [EOS_ID]
                if len(target) > self.seq_length / 4 * 3:
                    continue

                source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                try:
                    ATTR_LEN = 10
                    ATTR_NUM = 50
                    attr_keys = []
                    attr_values = []
                    json_obj = json.loads(tmp_list[4])
                    for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                        key = pair['property_name']
                        value = pair['values'][0]['value_name']
                        attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]
                        attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                        if len(attr_key) >= ATTR_LEN:
                            attr_key = attr_key[:ATTR_LEN]
                        else:
                            while len(attr_key) != ATTR_LEN:
                                attr_key.append(PAD_ID)
                        if len(attr_value) >= ATTR_LEN:
                            attr_value = attr_value[:ATTR_LEN]
                        else:
                            while len(attr_value) != ATTR_LEN:
                                attr_value.append(PAD_ID)

                        attr_keys.append(attr_key)
                        attr_values.append(attr_value)

                    if len(attr_keys) >= ATTR_NUM:
                        attr_keys = attr_keys[:ATTR_NUM]
                        attr_values = attr_values[:ATTR_NUM]
                    else:
                        while len(attr_keys) != ATTR_NUM:
                            attr_keys.append([PAD_ID] * ATTR_LEN)
                            attr_values.append([PAD_ID] * ATTR_LEN)
                except:
                    continue

                source = source[:int(self.seq_length / 4 - 1)]
                source += [DOS_ID]
                src_len = len(source)
                tgt_len = len(target)
                src = source + target[:-1]
                tgt = [0] * src_len + target[1:]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)


                item_id = tmp_list[0]


                pickle.dump((src, tgt, seg, src_len, tgt_len, prop_keys, prop_values, attr_keys, attr_values, item_id), f_write)

                if pos >= end - 1:
                    break

        f_write.close()





class OfficialStorylinepropattrpictDataset(data.Dataset):

    def __init__(self, args, vocab, tokenizer):
        import os
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.seq_length = args.seq_length
        self.length = int(os.popen('wc -l {}'.format(args.corpus_path)) \
                            .read().split()[0])

        self._tril_matrix = torch.tril(torch.ones((self.seq_length, self.seq_length), dtype=torch.long))

        #self.pict_path = os.path.join(os.path.split(args.corpus_path)[0], 'pict')
        self.pict_path = '/disk1/xinglin.hxl/data/pict'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = linecache.getline(self.corpus_path, index + 1)
        tmp_list = line.split('\t')
        if len(tmp_list) != 5:
            return self.__getitem__((index + 1) % self.length)

        target = [SOS_ID]
        word_list = []
        for pair in tmp_list[3].split("\001"):
            pair_list = pair.split("\002")
            word_list.append(pair_list[0])
        target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
        target += [EOS_ID]
        if len(target) > self.seq_length / 4 * 3:
            return self.__getitem__((index + 1) % self.length)

        source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

        PROP_LEN = 10
        PROP_NUM = 50
        prop_keys = []
        prop_values = []
        for pair in tmp_list[2].split("\001"):
            pair_list = pair.split("\002")
            prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
            prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

            if len(prop_key) >= PROP_LEN:
                prop_key = prop_key[:PROP_LEN]
            else:
                while len(prop_key) != PROP_LEN:
                    prop_key.append(PAD_ID)
            if len(prop_value) >= PROP_LEN:
                prop_value = prop_value[:PROP_LEN]
            else:
                while len(prop_value) != PROP_LEN:
                    prop_value.append(PAD_ID)

            prop_keys.append(prop_key)
            prop_values.append(prop_value)

        if len(prop_keys) >= PROP_NUM:
            prop_keys = prop_keys[:PROP_NUM]
            prop_values = prop_values[:PROP_NUM]
        else:
            while len(prop_keys) != PROP_NUM:
                prop_keys.append([PAD_ID] * PROP_LEN)
                prop_values.append([PAD_ID] * PROP_LEN)

        try:
            ATTR_LEN = 10
            ATTR_NUM = 50
            attr_keys = []
            attr_values = []
            json_obj = json.loads(tmp_list[4])
            for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                key = pair['property_name']
                value = pair['values'][0]['value_name']
                attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]
                attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                if len(attr_key) >= ATTR_LEN:
                    attr_key = attr_key[:ATTR_LEN]
                else:
                    while len(attr_key) != ATTR_LEN:
                        attr_key.append(PAD_ID)
                if len(attr_value) >= ATTR_LEN:
                    attr_value = attr_value[:ATTR_LEN]
                else:
                    while len(attr_value) != ATTR_LEN:
                        attr_value.append(PAD_ID)

                attr_keys.append(attr_key)
                attr_values.append(attr_value)

            if len(attr_keys) >= ATTR_NUM:
                attr_keys = attr_keys[:ATTR_NUM]
                attr_values = attr_values[:ATTR_NUM]
            else:
                while len(attr_keys) != ATTR_NUM:
                    attr_keys.append([PAD_ID] * ATTR_LEN)
                    attr_values.append([PAD_ID] * ATTR_LEN)
        except:
            return self.__getitem__((index + 1) % self.length)

        source = source[:int(self.seq_length / 4 - 1)]
        source += [DOS_ID]
        src_len = len(source)
        tgt_len = len(target)
        src = source + target[:-1]
        tgt = [0] * src_len + target[1:]
        seg = [1] * len(src)
        if len(src) >= self.seq_length:
            src = src[:self.seq_length]
            tgt = tgt[:self.seq_length]
            seg = seg[:self.seq_length]
        else:
            while len(src) != self.seq_length:
                src.append(PAD_ID)
                tgt.append(PAD_ID)
                seg.append(PAD_ID)

        item_id = tmp_list[0]


        ins = (src, tgt, seg, src_len, tgt_len, prop_keys, prop_values, attr_keys, attr_values, item_id)

        src = ins[0]
        tgt = ins[1]
        seg = ins[2]
        src_len = ins[3]
        tgt_len = ins[4] - 1
        mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
        mask[:, :src_len].fill_(1)
        second_st = src_len
        second_end = src_len + tgt_len
        mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        prop_keys = ins[5]
        prop_values = ins[6]

        attr_keys = ins[7]
        attr_values = ins[8]

        item_id = ins[9]
        pict_path = os.path.join(self.pict_path, item_id + '.png')
        if not os.path.exists(pict_path):
            #print(pict_path, 'does not exist!!')
            return self.__getitem__((index + 1) % self.length)

        pict = Image.open(pict_path)
        #pict = cv2.cvtColor(cv2.imread(pict_path), cv2.COLOR_BGR2RGB)
        pict = self.train_transform(pict)


        mask = mask.unsqueeze(0)
        mask = (1.0 - mask) * -10000

        return torch.LongTensor(src), \
            torch.LongTensor(tgt), \
            torch.LongTensor(seg), \
            torch.LongTensor(prop_keys), \
            torch.LongTensor(prop_values), \
            torch.LongTensor(attr_keys), \
            torch.LongTensor(attr_values), \
            pict, \
            mask






class OfficialStorylinepropattrpictmultiDataset(data.Dataset):

    def __init__(self, args, vocab, tokenizer):
        import os
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.seq_length = args.seq_length
        self.length = int(os.popen('wc -l {}'.format(args.corpus_path)) \
                            .read().split()[0])

        self._tril_matrix = torch.tril(torch.ones((self.seq_length, self.seq_length), dtype=torch.long))

        #self.pict_path = os.path.join(os.path.split(args.corpus_path)[0], 'pict')
        self.pict_path = '/disk1/xinglin.hxl/data/pict'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        ATTR_LEN = 10

        self.idx_2_attr_keys = []
        self.attr_keys_2_idx = {}
        self.all_attr_keys = []
        idx = 0
        with open('/home/xinglin.hxl/attr_all_key.txt') as file:
            for line in file:
                key = line.strip()
                if not key:
                    continue
                self.idx_2_attr_keys.append(key)
                self.attr_keys_2_idx[key] = idx
                idx += 1

                attr_key = [CLS_ID] + [vocab.get(w) for w in tokenizer.tokenize(key)]

                if len(attr_key) >= ATTR_LEN:
                    attr_key = attr_key[:ATTR_LEN]
                else:
                    while len(attr_key) != ATTR_LEN:
                        attr_key.append(PAD_ID)

                self.all_attr_keys.append(attr_key)

        self.idx_2_attr_values = []
        self.attr_values_2_idx = {}
        self.all_attr_values = []
        idx = 0
        with open('/home/xinglin.hxl/attr_all_value.txt') as file:
            for line in file:
                value = line.strip()
                if not value:
                    continue
                self.idx_2_attr_values.append(value)
                self.attr_values_2_idx[value] = idx
                idx += 1

                attr_value = [CLS_ID] + [vocab.get(w) for w in tokenizer.tokenize(value)]

                if len(attr_value) >= ATTR_LEN:
                    attr_value = attr_value[:ATTR_LEN]
                else:
                    while len(attr_value) != ATTR_LEN:
                        attr_value.append(PAD_ID)

                self.all_attr_values.append(attr_value)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = linecache.getline(self.corpus_path, index + 1)
        tmp_list = line.split('\t')
        if len(tmp_list) != 5:
            return self.__getitem__((index + 1) % self.length)

        target = [SOS_ID]
        word_list = []
        for pair in tmp_list[3].split("\001"):
            pair_list = pair.split("\002")
            word_list.append(pair_list[0])
        target += [self.vocab.get(w) for w in self.tokenizer.tokenize(','.join(word_list))]
        target += [EOS_ID]
        if len(target) > self.seq_length / 4 * 3:
            return self.__getitem__((index + 1) % self.length)

        source = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])] + [DOS_ID]

        PROP_LEN = 10
        PROP_NUM = 50
        prop_keys = []
        prop_values = []
        for pair in tmp_list[2].split("\001"):
            pair_list = pair.split("\002")
            prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
            prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

            if len(prop_key) >= PROP_LEN:
                prop_key = prop_key[:PROP_LEN]
            else:
                while len(prop_key) != PROP_LEN:
                    prop_key.append(PAD_ID)
            if len(prop_value) >= PROP_LEN:
                prop_value = prop_value[:PROP_LEN]
            else:
                while len(prop_value) != PROP_LEN:
                    prop_value.append(PAD_ID)

            prop_keys.append(prop_key)
            prop_values.append(prop_value)

        if len(prop_keys) >= PROP_NUM:
            prop_keys = prop_keys[:PROP_NUM]
            prop_values = prop_values[:PROP_NUM]
        else:
            while len(prop_keys) != PROP_NUM:
                prop_keys.append([PAD_ID] * PROP_LEN)
                prop_values.append([PAD_ID] * PROP_LEN)

        try:
            ATTR_LEN = 10
            ATTR_NUM = 50
            attr_keys_target = []
            attr_values_target = []
            attr_keys = []
            attr_values = []
            json_obj = json.loads(tmp_list[4])
            for pair in json_obj['data']['objects'][0]['properties_results'][0]:
                key = pair['property_name']
                value = pair['values'][0]['value_name']

                attr_keys_target.append(self.attr_keys_2_idx[key])
                attr_values_target.append(self.attr_values_2_idx[value])

                attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]
                attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                if len(attr_key) >= ATTR_LEN:
                    attr_key = attr_key[:ATTR_LEN]
                else:
                    while len(attr_key) != ATTR_LEN:
                        attr_key.append(PAD_ID)
                if len(attr_value) >= ATTR_LEN:
                    attr_value = attr_value[:ATTR_LEN]
                else:
                    while len(attr_value) != ATTR_LEN:
                        attr_value.append(PAD_ID)

                attr_keys.append(attr_key)
                attr_values.append(attr_value)

            if len(attr_keys_target) >= ATTR_NUM:
                attr_keys_target = attr_keys_target[:ATTR_NUM]
                attr_values_target = attr_values_target[:ATTR_NUM]
            else:
                while len(attr_keys_target) != ATTR_NUM:
                    attr_keys_target.append(0)
                    attr_values_target.append(0)

            if len(attr_keys) >= ATTR_NUM:
                attr_keys = attr_keys[:ATTR_NUM]
                attr_values = attr_values[:ATTR_NUM]
            else:
                while len(attr_keys) != ATTR_NUM:
                    attr_keys.append([PAD_ID] * ATTR_LEN)
                    attr_values.append([PAD_ID] * ATTR_LEN)
        except:
            return self.__getitem__((index + 1) % self.length)

        source = source[:int(self.seq_length / 4 - 1)]
        source += [DOS_ID]
        src_len = len(source)
        tgt_len = len(target)
        src = source + target[:-1]
        tgt = [0] * src_len + target[1:]
        seg = [1] * len(src)
        if len(src) >= self.seq_length:
            src = src[:self.seq_length]
            tgt = tgt[:self.seq_length]
            seg = seg[:self.seq_length]
        else:
            while len(src) != self.seq_length:
                src.append(PAD_ID)
                tgt.append(PAD_ID)
                seg.append(PAD_ID)

        item_id = tmp_list[0]


        ins = (src, tgt, seg, src_len, tgt_len, prop_keys, prop_values, attr_keys, attr_values, attr_keys_target, attr_values_target, item_id)

        src = ins[0]
        tgt = ins[1]
        seg = ins[2]
        src_len = ins[3]
        tgt_len = ins[4] - 1
        mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
        mask[:, :src_len].fill_(1)
        second_st = src_len
        second_end = src_len + tgt_len
        mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        prop_keys = ins[5]
        prop_values = ins[6]

        attr_keys = ins[7]
        attr_values = ins[8]

        attr_keys_target = ins[9]
        attr_values_target = ins[10]

        item_id = ins[11]
        pict_path = os.path.join(self.pict_path, item_id + '.png')
        if not os.path.exists(pict_path):
            #print(pict_path, 'does not exist!!')
            return self.__getitem__((index + 1) % self.length)

        pict = Image.open(pict_path)
        #pict = cv2.cvtColor(cv2.imread(pict_path), cv2.COLOR_BGR2RGB)
        pict = self.train_transform(pict)


        mask = mask.unsqueeze(0)
        mask = (1.0 - mask) * -10000

        return torch.LongTensor(src), \
            torch.LongTensor(tgt), \
            torch.LongTensor(seg), \
            torch.LongTensor(prop_keys), \
            torch.LongTensor(prop_values), \
            torch.LongTensor(attr_keys), \
            torch.LongTensor(attr_values), \
            torch.LongTensor(attr_keys_target), \
            torch.LongTensor(attr_values_target), \
            pict, \
            mask







class StorylinepropclsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                source = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                WORD_LEN = 10
                TARGET_NUM = 50
                target_words = []
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    target_word = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    if len(target_word) >= WORD_LEN:
                        target_word = target_word[:WORD_LEN]
                    else:
                        while len(target_word) != WORD_LEN:
                            target_word.append(PAD_ID)
                    target_words.append(target_word)

                if len(target_words) >= TARGET_NUM:
                    target_words = target_words[:TARGET_NUM]
                else:
                    while len(target_words) != TARGET_NUM:
                        target_words.append([PAD_ID] * WORD_LEN)

                source = source[:int(self.seq_length)]
                src_len = len(source)
                src = source
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, seg, src_len, prop_keys, prop_values, target_words), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StorylinepropclsrecurDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 3:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                source = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[0])]

                PROP_LEN = 10
                PROP_NUM = 50
                prop_keys = []
                prop_values = []
                for pair in tmp_list[1].split("\001"):
                    pair_list = pair.split("\002")
                    prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[1])]
                    prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]

                    if len(prop_key) >= PROP_LEN:
                        prop_key = prop_key[:PROP_LEN]
                    else:
                        while len(prop_key) != PROP_LEN:
                            prop_key.append(PAD_ID)
                    if len(prop_value) >= PROP_LEN:
                        prop_value = prop_value[:PROP_LEN]
                    else:
                        while len(prop_value) != PROP_LEN:
                            prop_value.append(PAD_ID)

                    prop_keys.append(prop_key)
                    prop_values.append(prop_value)

                if len(prop_keys) >= PROP_NUM:
                    prop_keys = prop_keys[:PROP_NUM]
                    prop_values = prop_values[:PROP_NUM]
                else:
                    while len(prop_keys) != PROP_NUM:
                        prop_keys.append([PAD_ID] * PROP_LEN)
                        prop_values.append([PAD_ID] * PROP_LEN)

                WORD_LEN = 10
                TARGET_NUM = 50
                target_words = []
                target_words.append([SOS_ID] + [PAD_ID] * (WORD_LEN - 1))
                for pair in tmp_list[2].split("\001"):
                    pair_list = pair.split("\002")
                    target_word = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair_list[0])]
                    if len(target_word) >= WORD_LEN:
                        target_word = target_word[:WORD_LEN]
                    else:
                        while len(target_word) != WORD_LEN:
                            target_word.append(PAD_ID)
                    target_words.append(target_word)
                target_words.append([EOS_ID] + [PAD_ID] * (WORD_LEN - 1))

                if len(target_words) >= TARGET_NUM:
                    target_words = target_words[:TARGET_NUM]
                else:
                    while len(target_words) != TARGET_NUM:
                        target_words.append([PAD_ID] * WORD_LEN)

                source = source[:int(self.seq_length)]
                src_len = len(source)
                src = source
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, seg, src_len, prop_keys, prop_values, target_words), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class StatDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        max_iter = 100000
        stat_dict = {}
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            idx = 0
            while True:
                if idx >= max_iter:
                    break
                try:
                    line = f.readline()
                    tmp_list = line.split('\t')
                    if len(tmp_list) != 2:
                        continue
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(tmp_list[1])]
                length = len(src)
                if length not in stat_dict:
                    stat_dict[length] = 0
                stat_dict[length] += 1

                idx += 1
                if pos >= end - 1:
                    break
        length_list_sorted = sorted(stat_dict.items(), key=lambda a:a[0])
        total_cnt = min(max_iter, end-start)
        curr_cnt = 0
        for pair in length_list_sorted:
            curr_cnt += pair[1]
            print(str(pair[0]) + "\t" + str(pair[1]) + "\t" + str(curr_cnt/total_cnt))
        f_write.close()


class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class Seq2seqDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            masks = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])
                src_len = ins[3]
                tgt_len = ins[4] - 1
                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))
            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg), \
                masks


class FpdgDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            src_type = []
            tgt = []
            tgt_type = []
            seg = []
            masks = []

            for ins in instances:
                src.append(ins[0])
                src_type.append(ins[1])
                tgt.append(ins[2])
                tgt_type.append(ins[3])
                seg.append(ins[4])
                src_len = ins[5]
                tgt_len = ins[6] - 1
                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))
            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(src), \
                torch.LongTensor(src_type), \
                torch.LongTensor(tgt), \
                torch.LongTensor(tgt_type), \
                torch.LongTensor(seg), \
                masks


class VaeDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            condition_title = []
            condition_text = []
            src = []
            tgt = []
            seg = []
            masks = []
            condition_title_seg = []
            condition_text_seg = []

            for ins in instances:
                condition_title.append(ins[0])
                condition_text.append(ins[1])
                src.append(ins[2])
                tgt.append(ins[3])
                seg.append(ins[4])
                src_len = ins[5]
                tgt_len = ins[6] - 1
                condition_title_seg.append(ins[7])
                condition_text_seg.append(ins[8])

                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))
            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(condition_title), \
                torch.LongTensor(condition_text), \
                torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg), \
                torch.LongTensor(condition_title_seg), \
                torch.LongTensor(condition_text_seg), \
                masks


class StorylinepropDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            masks = []
            prop_keys = []
            prop_values = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])
                src_len = ins[3]
                tgt_len = ins[4] - 1
                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))

                prop_keys.append(ins[5])
                prop_values.append(ins[6])

            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg), \
                torch.LongTensor(prop_keys), \
                torch.LongTensor(prop_values), \
                masks


class StorylinepropattrDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            masks = []
            prop_keys = []
            prop_values = []
            attr_keys = []
            attr_values = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])
                src_len = ins[3]
                tgt_len = ins[4] - 1
                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))

                prop_keys.append(ins[5])
                prop_values.append(ins[6])

                attr_keys.append(ins[7])
                attr_values.append(ins[8])

            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg), \
                torch.LongTensor(prop_keys), \
                torch.LongTensor(prop_values), \
                torch.LongTensor(attr_keys), \
                torch.LongTensor(attr_values), \
                masks


class StorylinepropattrpictDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length
        self._tril_matrix = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

        import os
        self.pict_path = os.path.join(os.path.split(dataset_path)[0], 'pict')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize])


    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            masks = []
            prop_keys = []
            prop_values = []
            attr_keys = []
            attr_values = []
            picts = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])
                src_len = ins[3]
                tgt_len = ins[4] - 1
                mask = torch.zeros(self.seq_length, self.seq_length, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
                masks.append(mask.view(1, self.seq_length, self.seq_length))

                prop_keys.append(ins[5])
                prop_values.append(ins[6])

                attr_keys.append(ins[7])
                attr_values.append(ins[8])

                item_id = ins[9]
                pict_path = os.path.join(self.pict_path, item_id + '.png')
                if not os.path.exists(pict_path):
                    print(pict_path, 'does not exist!!')
                    continue

                pict = Image.open(pict_path)
                pict = self.train_transform(pict)
                picts.append(pict.unsqueeze(0))


            masks = torch.cat(masks, 0)
            masks = masks.unsqueeze(1)
            masks = (1.0 - masks) * -10000

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg), \
                torch.LongTensor(prop_keys), \
                torch.LongTensor(prop_values), \
                torch.LongTensor(attr_keys), \
                torch.LongTensor(attr_values), \
                torch.cat(picts), \
                masks


class StorylinepropclsDataLoader(DataLoader):

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, seq_length, shuffle=False):
        DataLoader.__init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        self.seq_length = seq_length

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            seg = []
            prop_keys = []
            prop_values = []
            target_words = []

            for ins in instances:
                src.append(ins[0])
                seg.append(ins[1])
                src_len = ins[2]

                prop_keys.append(ins[3])
                prop_values.append(ins[4])
                target_words.append(ins[5])

            yield torch.LongTensor(src), \
                torch.LongTensor(seg), \
                torch.LongTensor(prop_keys), \
                torch.LongTensor(prop_values), \
                torch.LongTensor(target_words)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [SEP_ID]
                tgt_backward = [CLS_ID] + src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt_forward = tgt_forward[:self.seq_length]
                    tgt_backward = tgt_backward[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt_forward.append(PAD_ID)
                        tgt_backward.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt_forward, tgt_backward, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [CLS_ID] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3: # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [CLS_ID] + tokens_a + [SEP_ID]
                    src_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                    src_b = tokens_b + [SEP_ID]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.dup_factor = args.dup_factor

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    try:
                        f.readline()
                    except:
                        continue
                    finally:
                        pos += 1
                while True:
                    try:
                        line = f.readline()
                    except:
                        continue
                    finally:
                        pos += 1

                    src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]

                    if len(src) > self.seq_length:
                        src = src[:self.seq_length]
                    seg = [1] * len(src)

                    src, tgt = mask_seq(src, len(self.vocab))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        seg.append(PAD_ID)

                    pickle.dump((src, tgt, seg), f_write)

                    if pos >= end - 1:
                        break

        f_write.close()


class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue

            for ins in instances:
                src.append(ins[0])
                seg.append(ins[2])
                tgt.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt[-1][mask[0]] = mask[1]

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)