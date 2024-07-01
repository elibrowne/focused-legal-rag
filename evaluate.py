# evaluate.py

""" Remember to change the model name and save file name before running this code! """

import datasets
from ragatouille import RAGPretrainedModel

if __name__ == "__main__":
    # Load model for retrieval — an existing model name should be stored on disk.

    index_name = ".ragatouille/colbert/indexes/index_colbertv2.0base_gpu" # substitute with model being employed!
    model = RAGPretrainedModel.from_index(index_name)

    # Gather queries for evaluating retrieval. These can be passed to a model as an ordered list.

    qa_data = datasets.load_dataset("retrieval-bar/mbe", name="qa", split=datasets.Split.VALIDATION, trust_remote_code=True)
    # Remove items with no PID (not relevant to evaluating retrieval)
    qa_data = qa_data.filter(lambda entry : entry["gold_idx"] != "nan") # blanks are listed as "nan"
    # Concatenate prompt (when applicable) and question to get a list of queries
    queries = [(x[0] + " " + x[1]) if x[0] != "nan" else x[1] for x in zip(qa_data["prompt"], qa_data["question"])] # they were showing up as "nan Question..."; this fixed that
    gold_passage_ids = qa_data["gold_idx"] 
    print("Queries to search: " + str(len(queries)))

    # From here, we want to evaluate our results based on our query/gold passage pairs.
    # We can do this iteratively. Metrics in the "Legal Retrievers May..." paper for evaluating
    # BarQA included Recall@1, Recall@10, MRR@10, Recall@100, and Recall@1000.

    k = 1000 
    save_file = "evaluation_basecolbert.txt" # change this when testing different models
    retrieval_results = model.search(queries, k = k) 
    retrieved_1, retrieved_10, retrieved_100, retrieved_1000 = 0, 0, 0, 0 # count of queries that retrieved the gold passage @ k = 1, 10...
    mrr_total_10 = 0 # sum of MRR values @ k = 10 (for mean MRR calculation)

    for i, results in enumerate(retrieval_results):
        for j, result in enumerate(results):
            if result["document_id"] == gold_passage_ids[i]: # gold_idx[i] and query[i] (-> result[i]) should be the same example
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
