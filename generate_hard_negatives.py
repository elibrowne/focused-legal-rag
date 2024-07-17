import datasets 
from ragatouille.negative_miners import SimpleMiner

if __name__ == "__main__":
    # Collect pairs of (query, gold passage) tuples for training data
    qa_data = datasets.load_dataset("retrieval-bar/mbe", name="qa", split=datasets.Split.TRAIN, trust_remote_code=True)
    # Remove items with no PID (not useful for training because they have no gold passage)
    qa_data = qa_data.filter(lambda entry : entry["gold_idx"] != "nan") # blanks are listed as "nan"
    # Concatenate prompt (when applicable) and question to get a list of queries
    queries = [(x[0] + " " + x[1]) if x[0] != "nan" else x[1] for x in zip(qa_data["prompt"], qa_data["question"])] # they were showing up as "nan Question..."; this fixed that
    gold_passages = qa_data["gold_passage"]
    # gold_passage_ids = qa_data["gold_idx"] — likely unneeded for fine-tuning; only relevant in evaluation 
    zipped = zip(queries, gold_passages)
    data = list(zipped)

    miner = SimpleMiner(language_code = "en", model_size = "base")
    miner.build_index(save_index = True, save_path = "hard_negs")