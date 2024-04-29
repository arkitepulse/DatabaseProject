import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.word_embedding import WordEmbedding
from model.Predictors import AggPredictor,SelPredictor,CondPredictor
from library.table import Table
from library.query import Query
"""
    Custom Seq2Sql Model consisting of three individual models
    1. Aggregation predictor
    2. Selection Predictor
    3. Condition Predictor
"""
class MySeq2SQL(nn.Module):
   
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
                 use_gpu=False):
        super(MySeq2SQL, self).__init__()

        self.use_gpu = use_gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_count = 45
        self.max_tok_count = 200
        self.SQL_SYNTAX_TOKENS = [
            '<UNK>', '<END>', 'WHERE', 'AND',
            'EQL', 'GT', 'LT', '<BEG>'
        ]

        # Word embedding
        self.embed_layer = WordEmbedding(word_emb, N_word, use_gpu, self.SQL_SYNTAX_TOKENS, use_model=False)

        # Model for predicting aggregation clause
        self.agg_model = AggPredictor(N_word, N_h, N_depth)

        # Model for predicting select columns
        self.sel_model = SelPredictor(N_word, N_h, N_depth, self.max_tok_count)

        # Model for predicting the conditions
        self.cond_model = CondPredictor(N_word, N_h, N_depth, self.max_col_count, self.max_tok_count, use_gpu)

        # Loss function
        self.cross_entropy = nn.CrossEntropyLoss()
        if use_gpu:
            self.cuda()

    def generate_ground_truth_where_seq(self, question, columns, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(question, columns, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok + [',']]
            all_tokens = self.SQL_SYNTAX_TOKENS + connect_col + [None] + cur_q + [None]
            cur_seq = [all_tokens.index('<BEG>')]
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):]
                cur_seq = cur_seq + list(map(lambda tok: all_tokens.index(tok)
                                              if tok in all_tokens else 0, cur_where_query))
            cur_seq.append(all_tokens.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq

    def forward(self, question, columns, col_num, ground_truth_where=None, ground_truth_cond=None, ground_truth_sel=None):
        x_emb_var, x_len = self.embed_layer.gen_x_batch(question, columns)
        batch = self.embed_layer.gen_col_batch(columns)
        col_inp_var, col_name_len, col_len = batch

        agg_score = self.agg_model(x_emb_var, x_len)

        sel_score = self.sel_model(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        cond_score = self.cond_model(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, ground_truth_where, ground_truth_cond)

        return (agg_score, sel_score, cond_score)

    def calculate_loss(self, scores, truth_num, ground_truth_where):
        agg_score, sel_score, cond_score = scores
        loss = 0
        agg_truth = list(map(lambda x: x[0], truth_num))
        data = torch.from_numpy(np.array(agg_truth))
        if self.use_gpu:
            agg_truth_var = Variable(data.cuda())
        else:
            agg_truth_var = Variable(data)

        loss += self.cross_entropy(agg_score, agg_truth_var.long())

        sel_truth = list(map(lambda x: x[1], truth_num))
        data = torch.from_numpy(np.array(sel_truth))
        if self.use_gpu:
            sel_truth_var = Variable(data).cuda()
        else:
            sel_truth_var = Variable(data)

        loss += self.cross_entropy(sel_score, sel_truth_var.long())

        for b in range(len(ground_truth_where)):
            if self.use_gpu:
                cond_truth_var = Variable(torch.from_numpy(np.array(ground_truth_where[b][1:])).cuda())
            else:
                cond_truth_var = Variable(torch.from_numpy(np.array(ground_truth_where[b][1:])))
            cond_pred_score = cond_score[b, :len(ground_truth_where[b]) - 1]

            loss += (self.cross_entropy(
                cond_pred_score, cond_truth_var.long()) / len(ground_truth_where))

        return loss

    def evaluate_accuracy(self, predicted_queries, ground_truth_queries):
        total_err = agg_err = sel_err = cond_err = cond_num_err = \
            cond_col_err = cond_op_err = cond_val_err = 0.0
        for b, (pred_query, ground_truth_query) in enumerate(zip(predicted_queries, ground_truth_queries)):
            is_good = True

            agg_pred = pred_query['agg']
            agg_gt = ground_truth_query['agg']
            if agg_pred != agg_gt:
                agg_err += 1
                is_good = False

            sel_pred = pred_query['sel']
            sel_gt = ground_truth_query['sel']
            if sel_pred != sel_gt:
                sel_err += 1
                is_good = False

            cond_pred = pred_query['conds']
            cond_gt = ground_truth_query['conds']
            is_flag = True
            if len(cond_pred) != len(cond_gt):
                is_flag = False
                cond_num_err += 1

            if is_flag and set(
                    x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                is_flag = False
                cond_col_err += 1

            for idx in range(len(cond_pred)):
                if not is_flag:
                    break
                ground_truth_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if is_flag and cond_gt[ground_truth_idx][1] != cond_pred[idx][1]:
                    is_flag = False
                    cond_op_err += 1

            for idx in range(len(cond_pred)):
                if not is_flag:
                    break
                ground_truth_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if is_flag and str(cond_gt[ground_truth_idx][2]).lower() != \
                        str(cond_pred[idx][2]).lower():
                    is_flag = False
                    cond_val_err += 1

            if not is_flag:
                cond_err += 1
                is_good = False

            if not is_good:
                total_err += 1

        return np.array((agg_err, sel_err, cond_err)), total_err

    def generate_query(self, scores, question, columns, raw_question, raw_columns, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
                       '``': '"', '\'\'': '"', '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                        (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()
        agg_score, sel_score, cond_score = scores

        ret_queries = []
        B = len(cond_score)
        for b in range(B):
            cur_query = {}
            cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            cur_query['conds'] = []
            all_toks = self.SQL_SYNTAX_TOKENS + \
                       [x for toks in columns[b] for x in
                        toks + [',']] + [''] + question[b] + ['']
            cond_toks = []
            for where_score in cond_score[b].data.cpu().numpy():
                cond_tok = np.argmax(where_score)
                cond_val = all_toks[cond_tok]
                if cond_val == '<END>':
                    break
                cond_toks.append(cond_val)

            if verbose:
                print(cond_toks)
            if len(cond_toks) > 0:
                cond_toks = cond_toks[1:]
            st = 0
            while st < len(cond_toks):
                cur_cond = [None, None, None]
                ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
                    else cond_toks[st:].index('AND') + st
                if 'EQL' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('EQL') + st
                    cur_cond[1] = 0
                elif 'GT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('GT') + st
                    cur_cond[1] = 1
                elif 'LT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('LT') + st
                    cur_cond[1] = 2
                else:
                    op = st
                    cur_cond[1] = 0
                sel_col = cond_toks[st:op]
                to_idx = [x.lower() for x in raw_columns[b]]
                pred_col = merge_tokens(sel_col, raw_question[b] + ' || ' + \
                                        ' || '.join(raw_columns[b]))
                if pred_col in to_idx:
                    cur_cond[0] = to_idx.index(pred_col)
                else:
                    cur_cond[0] = 0
                cur_cond[2] = merge_tokens(cond_toks[op + 1:ed], raw_question[b])
                cur_query['conds'].append(cur_cond)
                st = ed + 1
            ret_queries.append(cur_query)

        return ret_queries

    
    
    def merge_tokens(token_list, raw_token_str):
        token_str = raw_token_str.lower()
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
        special_chars = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
                         '``': '"', '\'\'': '"', '--': u'\u2013'}
        result = ''
        double_quote_appear = 0
        for raw_token in token_list:
            if not raw_token:
                continue
            token = special_chars.get(raw_token, raw_token)
            if token == '"':
                double_quote_appear = 1 - double_quote_appear

            if len(result) == 0:
                pass
            elif len(result) > 0 and result + ' ' + token in token_str:
                result = result + ' '
            elif len(result) > 0 and result + token in token_str:
                pass
            elif token == '"':
                if double_quote_appear:
                    result = result + ' '
            elif token[0] not in alphabet:
                pass
            elif (result[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                    (result[-1] != '"' or not double_quote_appear):
                result = result + ' '
            result = result + token
        return result.strip()

    def save_readable_results(self, raw_question_seq, predicted_queries, ground_truth_queries, table_ids, table_data):
        file = open("./target_model_results.txt", "a+", encoding="utf-8")
        for index in range(len(predicted_queries)):
            pred_query_obj = Query.from_dict(predicted_queries[index])
            ground_truth_query_obj = Query.from_dict(ground_truth_queries[index])
            table_id = table_ids[index]
            table_info = table_data[table_id]
            table = Table(table_id, table_info["header"], table_info["types"], table_info["rows"])
            file.write(' '.join(raw_question_seq[index]))
            file.write("\n")
            file.write(table.query_str(ground_truth_query_obj))
            file.write("\n")
            file.write(table.query_str(pred_query_obj))
            file.write("\n\n")
        file.close()
