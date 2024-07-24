# see https://www.sbert.net/docs/pretrained-models/dpr.html

from sentence_transformers import SentenceTransformer, util
import datasets
import numpy as np

passage_encoder = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")

passage_data = datasets.load_dataset("retrieval-bar/mbe", name="passages", split=datasets.Split.VALIDATION, trust_remote_code=True)

def format(example):
    example["text"] = example["idx"] + " [SEP] " + example["text"]
    return example 

new_data = passage_data.map(format)
passages = new_data["text"]
ordered_ids = passage_data["idx"]

print(passages[0])
print("Finished formatting passages. Encoding passages...")

passage_embeddings = passage_encoder.encode(passages, show_progress_bar = True, device = "cuda:0")

print("Finished encoding. Loading query encoder...")

query_encoder = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")

qa_data = datasets.load_dataset("retrieval-bar/mbe", name="qa", split=datasets.Split.VALIDATION, trust_remote_code=True)
# Remove items with no PID (not relevant to evaluating retrieval)
qa_data = qa_data.filter(lambda entry : entry["gold_idx"] != "nan") # blanks are listed as "nan"
# Concatenate prompt (when applicable) and question to get a list of queries
queries = [(x[0] + " " + x[1]) if x[0] != "nan" else x[1] for x in zip(qa_data["prompt"], qa_data["question"])] # they were showing up as "nan Question..."; this fixed that
gold_passage_ids = qa_data["gold_idx"] 
print("Queries to search: " + str(len(queries)))

query_embeddings = [query_encoder.encode(q) for q in queries]
k = 1000 
save_file = "baseDPR_eval.txt" # change this when testing different models
retrieval_results = []

# Important: You must use dot-product, not cosine_similarity
for embedding in query_embeddings:
    scores = util.dot_score(embedding, passage_embeddings)
    retrieval_results.append(np.argsort(-scores)) # descending order not ascending order!!
    # next step: actually evaluate :) 

retrieved_1, retrieved_10, retrieved_100, retrieved_1000 = 0, 0, 0, 0 # count of queries that retrieved the gold passage @ k = 1, 10...
mrr_total_10 = 0 # sum of MRR values @ k = 10 (for mean MRR calculation)

for i, results in enumerate(retrieval_results):
    for j, result in enumerate(results.squeeze().tolist()):
        if ordered_ids[result] == gold_passage_ids[i]: # util.dot_score.argsort returns list of passage indexes, ordered_indices matches them to passage ID
            if j + 1 <= 1000: 
                retrieved_1000 += 1
                if j + 1 <= 100: 
                    retrieved_100 += 1
                    if j + 1 <= 10:
                        retrieved_10 += 1
                        mrr_total_10 += 1 / (j + 1)
                        if j + 1 == 1: retrieved_1 += 1
            break # break as soon as gold passage is found for efficiency

with open(save_file, 'a') as f:
    f.write("Results at K = " + str(k) + "\n")
    f.write(" - Recall (%) @ K = 1: " + str(100 * retrieved_1 / len(retrieval_results)) + "\n")
    f.write(" - Recall (%) @ K = 10: " + str(100 * retrieved_10 / len(retrieval_results)) + "\n")
    f.write(" - MRR (average, %): " + str(100 * mrr_total_10 / len(retrieval_results)) + "\n")
    f.write(" - Recall (%) @ K = 100: " + str(100 * retrieved_100 / len(retrieval_results)) + "\n")
    f.write(" - Recall (%) @ K = 1000: " + str(100 * retrieved_1000 / len(retrieval_results)) + "\n")
    f.write("\n")

