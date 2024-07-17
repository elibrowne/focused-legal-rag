import datasets 
from ragatouille import RAGTrainer

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

    # Load passage data (needed for mining hard negatives)
    passage_data = datasets.load_dataset("retrieval-bar/mbe", name="passages", split=datasets.Split.VALIDATION, trust_remote_code=True)
    passages = passage_data["text"]

    trainer = RAGTrainer(model_name = "colbert", pretrained_model_name = "colbert-ir/colbertv2.0") # this doesn't matter
    trainer.prepare_training_data(raw_data=data, all_documents=passages, data_out_path="./data/") # this should automatically mine hard negatives and save them if the custom install is working 