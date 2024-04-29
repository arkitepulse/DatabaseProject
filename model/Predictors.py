import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util.utils import run_lstm



class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth):
        super(AggPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2, num_layers=N_depth,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.att = nn.Linear(N_h, 1)
        self.out = nn.Sequential(nn.Linear(N_h, 6))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_emb_var, x_len):
        h_enc, _ = run_lstm(self.lstm, x_emb_var, x_len)
        att_val = self.att(h_enc).squeeze()
        max_len = max(x_len)
        att_val[:, max_len:] = -100
        att = self.softmax(att_val)
        K_agg = (h_enc * att.unsqueeze(2)).sum(1)
        agg_score = self.out(K_agg)
        return agg_score

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num):
        super(SelPredictor, self).__init__()
        self.max_tok_num = max_tok_num
        self.lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2, num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)
        self.att = nn.Linear(N_h, 1)
        self.col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                        num_layers=N_depth, batch_first=True,
                                        dropout=0.3, bidirectional=True)
        self.out_K = nn.Linear(N_h, N_h)
        self.out_col = nn.Linear(N_h, N_h)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()
        
    def col_name_encode(self,name_inp_var, name_l, col_l, enc_lstm):
        # Encode the columns.
        # The embedding of a column name is the last state of its LSTM output.
        name_h, _ = run_lstm(enc_lstm, name_inp_var, name_l)
        name_o = name_h[tuple(range(len(name_l))), name_l - 1]
        ret = torch.FloatTensor(len(col_l), max(col_l), name_o.size()[1]).zero_()
        if name_o.is_cuda:
            ret = ret.cuda()

        st = 0
        for idx, cur_l in enumerate(col_l):
            ret[idx, :cur_l] = name_o.data[st:st + cur_l]
            st += cur_l
        ret_var = Variable(ret)

        return ret_var, col_l

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num):
        max_x_len = max(x_len)

        e_col, _ = self.col_name_encode(col_inp_var, col_name_len, col_len, self.col_name_enc)

        h_enc, _ = run_lstm(self.lstm, x_emb_var, x_len)
        att_val = self.att(h_enc).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)
        K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_sel_expand = K_sel.unsqueeze(1)

        sel_score = self.out(self.out_K(K_sel_expand) + self.out_col(e_col)).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score

class CondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu):
        super(CondPredictor, self).__init__()
        print("Seq2SQL where prediction")
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu

        self.lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                 num_layers=N_depth, batch_first=True,
                                 dropout=0.3, bidirectional=True)
        self.decoder = nn.LSTM(input_size=self.max_tok_num,
                                    hidden_size=N_h, num_layers=N_depth,
                                    batch_first=True, dropout=0.3)

        self.out_g = nn.Linear(N_h, N_h)
        self.out_h = nn.Linear(N_h, N_h)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()

    def gen_gt_batch(self, tok_seq, gen_inp=True):
        B = len(tok_seq)
        ret_len = np.array([len(seq) - 1 for seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros((B, max_len, self.max_tok_num), dtype=np.float32)
        for b, seq in enumerate(tok_seq):
            out_seq = seq[:-1] if gen_inp else seq[1:]
            for t, tok_id in enumerate(out_seq):
                ret_array[b, t, tok_id] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        return Variable(ret_inp), ret_len

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_where, gt_cond):
        max_x_len = max(x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.lstm, x_emb_var, x_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]), dim=2)
                               for hid in hidden)
        if gt_where is not None:
            gt_seq, gt_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.decoder, gt_seq, gt_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)
            cond_score = self.out(self.out_h(h_enc_expand) + self.out_g(g_s_expand)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    cond_score[idx, :, num:] = -100
        else:
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0, 7] = 1  # Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                cur_cond_score = self.out(self.out_h(h_enc_expand) +
                                               self.out_g(g_s_expand)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_score[b, num:] = -100
                scores.append(cur_cond_score)

                _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
                ans_tok_var = ans_tok_var.unsqueeze(1)

                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  # To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  # Find the <END> token
                        done_set.add(idx)
                t += 1

            cond_score = torch.stack(scores, 1)

        return cond_score
