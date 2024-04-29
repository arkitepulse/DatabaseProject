import json
import torch
from torch import nn as nn
from torch.autograd import Variable
from library.dbengine import DBEngine
import numpy as np


def load_dataset(custom_dataset_name):
    print("Loading dataset:", custom_dataset_name)
    sql_data_path = 'Data/data/tokenized_' + custom_dataset_name + '.jsonl'
    table_data_path = 'Data/data/tokenized_' + custom_dataset_name + '.tables.jsonl'
    db_file_path = 'Data/data/' + custom_dataset_name + '.db'

    loaded_sql_data = []
    loaded_table_data = {}
    
    with open(sql_data_path, encoding="utf-8") as sql_file:
        for line in sql_file:
            sql_entry = json.loads(line.strip())
            loaded_sql_data.append(sql_entry)

    with open(table_data_path, encoding="utf-8") as table_file:
        for line in table_file:
            table_entry = json.loads(line.strip())
            loaded_table_data[table_entry[u'id']] = table_entry

    return loaded_sql_data, loaded_table_data, db_file_path

def best_model_name():
    
    best_sel_model = 'saved_model/seq2sql.sel_model'
    best_cond_model = 'saved_model/seq2sql.cond_'
    best_agg_model = 'saved_model/seq2sql.agg_model'
    return best_agg_model, best_sel_model, best_cond_model


def generate_batch_sequence(sql_data, table_data, idxes, start, end):
    question_sequence = []
    column_sequence = []
    number_of_columns = []
    answer_sequence = []
    query_sequence = []
    ground_truth_condition_sequence = []
    raw_data = []
    for i in range(start, end):
        sql = sql_data[idxes[i]]
        question_sequence.append(sql['tokenized_question'])
        column_sequence.append(table_data[sql['table_id']]['tokenized_header'])
        number_of_columns.append(len(table_data[sql['table_id']]['header']))
        answer_sequence.append((sql['sql']['agg'],
                        sql['sql']['sel'],
                        len(sql['sql']['conds']),
                        tuple(x[0] for x in sql['sql']['conds']),
                        tuple(x[1] for x in sql['sql']['conds'])))
        query_sequence.append(sql['tokenized_query'])
        ground_truth_condition_sequence.append(sql['sql']['conds'])
        raw_data.append((sql['question'], table_data[sql['table_id']]['header'], sql['query']))


    return question_sequence, column_sequence, number_of_columns, answer_sequence, query_sequence,\
        ground_truth_condition_sequence, raw_data


def generate_batch_query(sql_data, idxes, start, end):
    query_gt = []
    table_ids = []
    for i in range(start, end):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids



def epoch_exec_acc(model_instance, batch_size_val, sql_data, table_data, db_path):
    engine_instance = DBEngine(db_path)

    model_instance.eval()
    perm_indices = list(range(len(sql_data)))
    total_acc_num = 0.0
    start_index = 0
    while start_index < len(sql_data):
        end_index = start_index + batch_size_val if start_index + batch_size_val < len(perm_indices) else len(perm_indices)
        q_sequence, col_sequence, col_num, ans_sequence, query_sequence, ground_truth_cond_sequence, raw_data = \
            generate_batch_sequence(sql_data, table_data, perm_indices, start_index, end_index)
        raw_q_sequence = [x[0] for x in raw_data]
        raw_col_sequence = [x[1] for x in raw_data]
        query_gt, table_ids = generate_batch_query(sql_data, perm_indices, start_index, end_index)
        ground_truth_sel_sequence = [x[1] for x in ans_sequence]
        score_output = model_instance.forward(q_sequence, col_sequence, col_num, ground_truth_sel=ground_truth_sel_sequence)
        pred_queries = model_instance.generate_query(score_output, q_sequence, col_sequence, raw_q_sequence, raw_col_sequence)

        for idx_val, (sql_gt_val, sql_pred_val, tid_val) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt_val = engine_instance.execute(tid_val,
                                    sql_gt_val['sel'], sql_gt_val['agg'], sql_gt_val['conds'])
            try:
                ret_pred_val = engine_instance.execute(tid_val,
                                          sql_pred_val['sel'], sql_pred_val['agg'], sql_pred_val['conds'])
            except:
                ret_pred_val = None
            total_acc_num += (ret_gt_val == ret_pred_val)

        start_index = end_index

    return total_acc_num / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, save_results=False):
    model.eval()
    perm = list(range(len(sql_data)))
    start = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while start < len(sql_data):
        end = start + batch_size if start + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data =\
            generate_batch_sequence(sql_data, table_data, perm, start, end)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = generate_batch_query(sql_data, perm, start, end)
        ground_truth_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, ground_truth_sel=ground_truth_sel_seq)
        pred_queries = model.generate_query(score, q_seq, col_seq,
                                       raw_q_seq, raw_col_seq)
        one_err, tot_err = model.evaluate_accuracy(pred_queries, query_gt)
        
        if save_results:
            model.save_readable_results(q_seq,pred_queries, query_gt, table_ids, table_data)

        one_acc_num += (end - start - one_err)
        tot_acc_num += (end - start - tot_err)

        start = end
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def run_lstm(lstm_model, input_data, input_lengths, hidden_state=None):
    sorted_indices = np.array(sorted(range(len(input_lengths)), key=lambda k: input_lengths[k], reverse=True))
    sorted_lengths = input_lengths[sorted_indices]
    inverse_indices = np.argsort(sorted_indices)
    if input_data.is_cuda:
        sorted_indices = torch.LongTensor(sorted_indices).cuda()
        inverse_indices = torch.LongTensor(inverse_indices).cuda()

    lstm_input = nn.utils.rnn.pack_padded_sequence(input_data[sorted_indices],
                                                    sorted_lengths, batch_first=True)
    if hidden_state is None:
        lstm_hidden_state = None
    else:
        lstm_hidden_state = (hidden_state[0][:, sorted_indices], hidden_state[1][:, sorted_indices])

    sorted_output_seq, sorted_hidden_state = lstm_model(lstm_input, lstm_hidden_state)
    output_seq = nn.utils.rnn.pad_packed_sequence(sorted_output_seq, batch_first=True)[0][inverse_indices]
    hidden_state_output = (sorted_hidden_state[0][:, inverse_indices], sorted_hidden_state[1][:, inverse_indices])
    return output_seq, hidden_state_output

