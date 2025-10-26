import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_sci_scibert", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="model/all-mpnet-base-v2", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="medical", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    return parser.parse_args()


def load_dataset(dataset_name,tokenizer): 
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model):
    embedding_model = SentenceTransformer(embedding_model,device="cuda")
    return embedding_model

def main():
    args = parse_arguments()
    embedding_model = load_embedding_model(args.embedding_model)
    questions,passages = load_dataset(args.dataset_name)
    setup_logging(f"results/{args.dataset_name}/log.txt")
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model
    )
    rag_model = LinearRAG(global_config=config)
    rag_model.index(passages)
    questions = rag_model.qa(questions)
    os.makedirs(f"results/{args.dataset_name}", exist_ok=True)
    with open(f"results/{args.dataset_name}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{args.dataset_name}/predictions.json")
    evaluator.evaluate(max_workers=args.max_workers)
if __name__ == "__main__":
    main()