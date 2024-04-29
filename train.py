from util.utils import *
from model.seq2sql import MySeq2SQL
from util.constants import *
import nltk
from library.table import Table
from library.query import Query
import records

def load_w_embeddings(file_name):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name, encoding="utf-8") as inf:
        for idx, line in enumerate(inf):
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret

def train_epoch(model, optimizer, batch_size, sql_data, table_data):
    model.train()
    perm = np.random.permutation(len(sql_data))
    cumulative_loss = 0.0
    start = 0
    while start < len(sql_data):
        end = min(start + batch_size, len(perm))

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data = \
            generate_batch_sequence(sql_data, table_data, perm, start, end)
        ground_truth_where_seq = model.generate_ground_truth_where_seq(q_seq, col_seq, query_seq)
        ground_truth_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, ground_truth_where=ground_truth_where_seq,
                              ground_truth_cond=ground_truth_cond_seq, ground_truth_sel=ground_truth_sel_seq)
        loss = model.calculate_loss(score, ans_seq, ground_truth_where_seq)
        cumulative_loss += loss.data.cpu().numpy() * (end - start)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        start = end

    return cumulative_loss / len(sql_data)

def train_model_seq2sql():
    
    train_sql_data, train_table_data, TRAIN_DB = load_dataset("train")
    dev_sql_data, dev_table_data, DEV_DB = load_dataset("dev")

    
    word_emb = load_w_embeddings('Data/glove/glove.6B.300d.txt')

    
    model = MySeq2SQL(word_emb, N_word=300, use_gpu=GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   
    agg_model, sel_model, cond_model = best_model_name()

    
    init_acc = epoch_acc(model, BATCH_SIZE, dev_sql_data, dev_table_data)
    best_agg_acc = init_acc[1][0]
    best_sel_acc = init_acc[1][1]
    best_cond_acc = init_acc[1][2]

    
    torch.save(model.sel_model.state_dict(), sel_model)
    torch.save(model.agg_model.state_dict(), agg_model)
    torch.save(model.cond_model.state_dict(), cond_model)

    
    epoch_losses = []

    for epoch in range(TRAINING_EPOCHS):
        print('Epoch:', epoch + 1)
        
        
        epoch_loss = train_epoch(model, optimizer, BATCH_SIZE, train_sql_data, train_table_data)
        epoch_losses.append(epoch_loss)
        print('Loss =', epoch_loss)

        
        train_accuracy = epoch_acc(model, BATCH_SIZE, train_sql_data, train_table_data)
        print('Train accuracy:', train_accuracy)
        
        dev_accuracy = epoch_acc(model, BATCH_SIZE, dev_sql_data, dev_table_data)
        print('Dev accuracy:', dev_accuracy)
        
        # Update best models if accuracy improves
        if dev_accuracy[1][0] > best_agg_acc:
            best_agg_acc = dev_accuracy[1][0]
            torch.save(model.agg_model.state_dict(), agg_model)
        if dev_accuracy[1][1] > best_sel_acc:
            best_sel_acc = dev_accuracy[1][1]
            torch.save(model.sel_model.state_dict(), sel_model)
        if dev_accuracy[1][2] > best_cond_acc:
            best_cond_acc = dev_accuracy[1][2]
            torch.save(model.cond_model.state_dict(), cond_model)

        print('Best val accuracy:', (best_agg_acc, best_sel_acc, best_cond_acc))
        
class DataConverter:
    def __init__(self):
        self.tables_data = {}

    def build_table_mapping(self, dataset):
        tables_df = pd.read_json(f"Data/data/{dataset}.tables.jsonl", lines=True)
        data = pd.DataFrame()
        for _, line in tables_df.iterrows():
            line["tokenized_header"] = [self.tokenize_document(column) for column in line["header"]]
            line_df = pd.DataFrame(line).transpose()
            data = data.append(line_df)
            self.tables_data[line["id"]] = line
        self.save_dataframe(data, f"Data/data/tokenized_{dataset}.tables.jsonl")

    def get_query_and_table(self, json_line):
        q = Query.from_dict(json_line["sql"])
        table = self.tables_data[json_line["table_id"]]
        return table, q

    @staticmethod
    def execute_query(table, query):
        db = records.Database('sqlite:///../Data/data/train.db')
        conn = db.get_connection()
        query, result = table.execute_query(conn, query)
        conn.close()
        print(query, result)

    @staticmethod
    def tokenize_document(doc):
        operators = {'=': 'EQL', '>': 'GT', '<': 'LT'}
        syntax_tokens = ["SELECT", "COUNT", "WHERE", "AND", "OR", "FROM"]
        tokens = nltk.word_tokenize(doc)
        return [operators[token] if token in operators else token.lower() for token in tokens if token not in syntax_tokens]

    @staticmethod
    def save_dataframe(data, filename):
        data.to_json(filename, orient='records', lines=True)

    def build_tokenized_dataset(self, dataset):
        self.build_table_mapping(dataset)
        queries_df = pd.read_json(f"Data/data/{dataset}.jsonl", lines=True)
        data = pd.DataFrame()
        for _, line in queries_df.iterrows():
            table, query = self.get_query_and_table(line)
            query_str = table.query_str(query)
            tokenized_query = self.tokenize_document(query_str)
            line["tokenized_query"] = tokenized_query
            line["query"] = " ".join(tokenized_query)
            line["tokenized_question"] = self.tokenize_document(line["question"])
            line_df = pd.DataFrame(line).transpose()
            data = data.append(line_df)
        self.save_dataframe(data, f"Data/data/tokenized_{dataset}.jsonl")
        
#eck to perform tokenisation only if files are not present , also to download source data sets
data_conv = DataConverter()
for dataset in ["train", "dev", "test"]:
    data_conv.build_tokenized_dataset(dataset)
        
if __name__ == '__main__':
    train_seq2sql()