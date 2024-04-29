from util.utils import *
from util.constants import *
from model.seq2sql import MySeq2SQL

def load_w_embeddings(file_name):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name, encoding="utf-8") as inf:
        for idx, line in enumerate(inf):
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret

def test_seq2sql():
    test_sql_data, test_table_data, TEST_DB = load_dataset("test")

    # load glove word embeddings and initialize the model
    word_emb = load_w_embeddings('Data/glove/glove.6B.300d.txt')
    model = MySeq2SQL(word_emb, N_word=300, use_gpu=GPU)

    # Load the best model state saved during training
    agg_m, sel_m, cond_m = best_model_name()
    model.agg_model.load_state_dict(torch.load(agg_m))
    model.sel_model.load_state_dict(torch.load(sel_m))
    model.cond_model.load_state_dict(torch.load(cond_m))

    # Run the model on the test data and get the logical accuracy
    logical_accuracy_score =\
        epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, save_results = True)

    # Run the model on the test data and get the execution accuracy
    execution_accuracy_score =\
        epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB)
    
    print("Test logical accuracy: %s;\n  breakdown on (agg, sel, where): %s" % logical_accuracy_score)
    print("Test execution accuracy: %s" % execution_accuracy_score)


if __name__ == '__main__':
    test_seq2sql()
