import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, emb_size, gpu, syntax_tokens, use_model):
        super(WordEmbedding, self).__init__()
        self.emb_size = emb_size
        self.use_model = use_model
        self.gpu = gpu
        self.syntax_tokens = syntax_tokens
        self.word_emb = word_emb

    def gen_x_batch(self, questions, columns):
        B = len(questions)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (q, col) in enumerate(zip(questions, columns)):
            q_val = list(map(lambda x: self.word_emb.get(x, np.zeros(self.emb_size, dtype=np.float32)), q))
            if self.use_model:
                val_embs.append([np.zeros(self.emb_size, dtype=np.float32)] + q_val + [
                    np.zeros(self.emb_size, dtype=np.float32)])  # <BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                col_val = list(map(lambda x: self.word_emb.get(x, np.zeros(self.emb_size, dtype=np.float32)),
                                   [x for toks in col for x in toks + [',']]))
                val_embs.append([np.zeros(self.emb_size, dtype=np.float32) for _ in self.syntax_tokens] + col_val + [
                    np.zeros(self.emb_size, dtype=np.float32)] + q_val + [np.zeros(self.emb_size, dtype=np.float32)])
                val_len[i] = len(self.syntax_tokens) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.emb_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, val_len
    
    def str_list_to_batch(self, str_list):
        B = len(str_list)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            val = [self.word_emb.get(x, np.zeros(self.emb_size, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.emb_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)
        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len
