# generate_index_and_evaluate.py

# RAGatouille creates an index using a pretrained model and then runs
# queries on that index (which has the model built-in). This script generates
# those indices for any pretrained model. It also adds document IDs to the 
# index for ease of use during evaluation. This script evaluates on the TEST set.

import datasets
from ragatouille import RAGPretrainedModel

model_name = "colbert-ir/colbertv2.0" # substitute with model being employed!
model = RAGPretrainedModel.from_pretrained(model_name)

passage_data = datasets.load_dataset("retrieval-bar/mbe", name="passages", split=datasets.Split.TEST, trust_remote_code=True)
passages, passage_ids = passage_data["text"], passage_data["idx"]
# The index is stored locally: it's named based on the model it used; it uses passage IDs as document IDs. 
index_path = model.index(index_name="index_"+model_name, collection=passages, document_ids=passage_ids)

# From here, we want to evaluate our results based on our query/gold passage pairs.
# We can do this iteratively. Metrics in the "Legal Retrievers May..." paper for evaluating
# BarQA included Recall@1, Recall@10, MRR@10, Recall@100, and Recall@1000.

qa_data = datasets.load_dataset("retrieval-bar/mbe", name="qa", split=datasets.Split.TEST, trust_remote_code=True)
# Remove items with no PID (not relevant to evaluating retrieval)
qa_data = qa_data.filter(lambda entry : entry["gold_idx"] != "nan") # blanks are listed as "nan"
# Concatenate prompt (when applicable) and question to get a list of queries
queries = [(x[0] + " " + x[1]) if x[0] != "nan" else x[1] for x in zip(qa_data["prompt"], qa_data["question"])] # they were showing up as "nan Question..."; this fixed that
gold_passages = qa_data["gold_idx"] # list of gold passage IDs (aligned with document IDs of index)

# This method calculates evaluation metrics @ K and adds them to the 
def metrics(k, results, gold_passage_id, add_mrr=False):
    pass 

metrics(1, ?, ?)
metrics(10, ?, ?, True) # also calculating MRR @ 10
metrics(100, ?, ?)
metrics(1000, ?, ?)