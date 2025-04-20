import json
import re
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from unsloth import FastLanguageModel

from config import DATA_DIR, logger
from src.embeddings import CustomHuggingFaceEmbeddings


# === GPU Detection ===
def is_old_gpu():
    if not torch.cuda.is_available():
        return True
    gpu_name = torch.cuda.get_device_name().lower()
    unsupported_keywords = ["p100", "k80", "m60", "t4"]
    return any(keyword in gpu_name for keyword in unsupported_keywords)


USE_VLLM = not is_old_gpu()

if USE_VLLM:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=4096)
    sampling_params_short = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=64)
    sampling_params_medium = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=256)
    sampling_params_long = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=512)
    logger.info("Using vLLM backend.")
else:
    sampling_params = {"temperature": 0.3, "top_p": 0.95, "max_new_tokens": 4096}
    sampling_params_short = {"temperature": 0.3, "top_p": 0.95, "max_new_tokens": 64}
    sampling_params_medium = {"temperature": 0.3, "top_p": 0.95, "max_new_tokens": 256}
    sampling_params_long = {"temperature": 0.3, "top_p": 0.95, "max_new_tokens": 512}
    logger.warning("Old GPU detected. Falling back to Unsloth/Transformers inference.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=USE_VLLM,
    gpu_memory_utilization=0.6,
)


# === Generation Utility ===
def call_fast_generate(prompts, params):
    if USE_VLLM:
        return model.fast_generate(prompts, sampling_params=params)
    else:
        return model.fast_generate(
            prompts,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_new_tokens=params["max_new_tokens"],
        )


# Load and split markdown
docs = UnstructuredMarkdownLoader("./data/mission_report.md").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

chunks_df = pd.DataFrame(
    {
        "chunk_id": range(1, len(chunks) + 1),
        "content": [chunk.page_content for chunk in chunks],
        "metadata": [chunk.metadata for chunk in chunks],
    }
)
chunks_df.to_csv(DATA_DIR / "chunks.csv", index=False)
logger.info(f"Saved {len(chunks)} chunks to {DATA_DIR}/chunks.csv")

embeddings = CustomHuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(str(DATA_DIR))
logger.info(f"Saved FAISS index to {DATA_DIR}")

# === Paraphrasing ===
PARAPHRASE_PROMPTS = [
    """Rewrite this text in a formal, scholarly tone. Keep it very concise - summarize in 1-2 short sentences. Only output the paraphrased text:\n\nTEXT: {text}""",
    """Rewrite this text in a clear, simple way that's easy to understand. Provide a medium-length explanation with key details. Only output the paraphrased text:\n\nTEXT: {text}""",
    """Rewrite this text in a vivid, engaging style. Expand on the details and provide a comprehensive, detailed version. Only output the paraphrased text:\n\nTEXT: {text}""",
]


def generate_paraphrases(text: str) -> list:
    responses = []
    sampling_params_list = [
        sampling_params_short,
        sampling_params_medium,
        sampling_params_long,
    ]

    for prompt_template, params in zip(PARAPHRASE_PROMPTS, sampling_params_list):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_template.format(text=text)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        output = call_fast_generate([formatted_prompt], params)
        responses.append(output[0].outputs[0].text)
    return responses


logger.info("Paraphrasing chunks and adding to vector store...")
all_paraphrased = []
for i, chunk in enumerate(chunks):
    paragraphs = [p.strip() for p in chunk.page_content.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        paraphrased_versions = generate_paraphrases(paragraph)
        for version in paraphrased_versions:
            all_paraphrased.append(
                {
                    "chunk_id": i + 1,
                    "original_paragraph": paragraph,
                    "paraphrased_text": version,
                }
            )

paraphrased_df = pd.DataFrame(all_paraphrased)
paraphrased_df.to_csv(DATA_DIR / "paragraphs_noise.csv", index=False)
logger.info(f"Saved {len(all_paraphrased)} paraphrased paragraphs")

paraphrased_docs = [
    Document(
        page_content=item["paraphrased_text"],
        metadata={"chunk_id": item["chunk_id"], "is_paraphrase": True},
    )
    for item in all_paraphrased
]

batch_size = 100
paraphrased_vectorstore = None
for i in range(0, len(paraphrased_docs), batch_size):
    batch = paraphrased_docs[i : i + batch_size]
    batch_vectorstore = FAISS.from_documents(batch, embeddings)
    if paraphrased_vectorstore is None:
        paraphrased_vectorstore = batch_vectorstore
    else:
        paraphrased_vectorstore.merge_from(batch_vectorstore)

if paraphrased_vectorstore is not None:
    vectorstore.merge_from(paraphrased_vectorstore)
    vectorstore.save_local(str(DATA_DIR))
    logger.info("Saved updated FAISS index with paraphrased documents")


# === QA Generation ===
def batch_generate(prompts: list) -> list:
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    outputs = call_fast_generate(formatted, sampling_params)
    return [output.outputs[0].text for output in outputs]


def parse_qa_block(block: str):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    question, answer, difficulty = None, None, None
    for line in lines:
        lower = line.lower()
        if question is None and lower.startswith("question:"):
            question = line[len("question:") :].strip()
        elif answer is None and lower.startswith("answer:"):
            answer = line[len("answer:") :].strip()
        elif difficulty is None and lower.startswith("difficulty:"):
            difficulty = line[len("difficulty:") :].strip()
    if question and answer and difficulty:
        return question, answer, difficulty
    if len(lines) == 3:
        return lines[0], lines[1], lines[2]
    return None


def parse_multiple_qa_output(output: str) -> list:
    blocks = re.split(r"\n\s*\n", output.strip())
    return [parse_qa_block(block) for block in blocks if parse_qa_block(block)]


def generate_question_batch_for_chunks(chunks: list, num_questions: int = 2) -> list:
    prompts, chunk_ids, chunk_contents = [], [], []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        prompt = (
            f"You are a question generator. Generate {num_questions} questions based on the following text.\n"
            "Rules:\n"
            "1. Questions must be answerable using ONLY the information in the text\n"
            "2. Answers must be directly stated in the text\n"
            "3. Each question should test understanding of a different aspect of the text\n"
            "4. Questions should be clear and specific\n"
            "5. Answers should be concise and factual\n\n"
            "For each QA pair, output exactly three lines with no extra commentary:\n"
            "Line 1: Question: <your question>\n"
            "Line 2: Answer: <the answer>\n"
            "Line 3: Difficulty: <easy, medium, or hard>\n"
            "Do not include any additional text.\n\n"
            f"Text:\n{content}\n"
        )
        prompts.append(prompt)
        chunk_ids.append(i + 1)
        chunk_contents.append(content)

    outputs = batch_generate(prompts)
    final_questions = []
    for i, output in enumerate(outputs):
        qa_pairs = parse_multiple_qa_output(output)
        if not qa_pairs:
            continue
        valid = [
            (q, a, d) for q, a, d in qa_pairs if a.lower() in chunk_contents[i].lower()
        ]
        for q, a, d in valid[:num_questions]:
            supporting = [
                p.strip() for p in chunk_contents[i].split("\n\n") if p.strip()
            ]
            final_questions.append(
                {
                    "id": str(chunk_ids[i]),
                    "question": q,
                    "answer": a,
                    "supporting_paragraphs": supporting,
                }
            )
    logger.info(f"Generated {len(final_questions)} valid QA pairs")
    return final_questions


logger.info("Generating question-answer pairs...")
all_questions = generate_question_batch_for_chunks(chunks, num_questions=2)
questions_path = DATA_DIR / "questions.jsonl"
with open(questions_path, "w") as f:
    for q in all_questions:
        f.write(json.dumps(q) + "\n")
logger.info(f"Saved questions to {questions_path}")
