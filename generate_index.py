# generate_index_and_evaluate.py

""" Remember to change the model name and index name before running this code! """

# RAGatouille creates an index using a pretrained model and then runs
# queries on that index (which has the model built-in). This script generates
# those indices for any pretrained model. It also adds document IDs to the 
# index for ease of use during evaluation. This script evaluates on the TEST set.

import datasets
from ragatouille import RAGPretrainedModel
import faiss

# RAGatouille requires scripts to be run from main.
if __name__ == "__main__":
    model_name = "colbert-ir/colbertv2.0" # substitute with model being employed!
    model = RAGPretrainedModel.from_pretrained(model_name)

    passage_data = datasets.load_dataset("retrieval-bar/mbe", name="passages", split=datasets.Split.VALIDATION, trust_remote_code=True)
    passages, passage_ids = passage_data["text"], passage_data["idx"]
    # The index is stored locally: it's named based on the model it used; it uses passage IDs as document IDs. 
    index_path = model.index(index_name="index_colbertv2.0base_gpu", collection=passages, document_ids=passage_ids, bsize=8, max_document_length=512) # document length maximized for BERT
