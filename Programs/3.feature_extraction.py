import json, faiss, numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX = "Output/pubmed_index.faiss"
META = "Output/pubmed_meta.jsonl"
OUT  = "Output/features_pubmed_meta.jsonl"
K_RETRIEVE = 6

# Load OpenAI API
with open("APIs_dictionary.json") as f:
    keys = json.load(f)
client = OpenAI(api_key=keys.get("OpenAI"))

# Local LLM for fallback
local_llm = pipeline(
    "text-generation",
    model="NousResearch/Hermes-2-Pro-Llama-3-8B",
    device_map="auto",
    torch_dtype="auto"
)

# Embedding encoder fallback (local)
encoder = SentenceTransformer("all-roberta-large-v1")

# Load metadata and index
def load_meta():
    with open(META, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
def load_index():
    return faiss.read_index(INDEX)

# Prompt embeddings & Retrieve context via RAG
def retrieve_context(query, index, meta, k=K_RETRIEVE):
    try:
        r = client.embeddings.create(model="text-embedding-3-large", input=query)
        qv = np.array(r.data[0].embedding, dtype="float32").reshape(1, -1)
    except Exception:
        qv = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    if qv.shape[1] != index.d:
        qv = np.pad(qv, ((0, 0), (0, max(0, index.d - qv.shape[1]))))[:, :index.d]
    _, ids = index.search(qv, k)
    return " ".join(f"{meta[i].get('title','')} {meta[i].get('abstract','')}" for i in ids[0])

# LLM-based feature extraction with RAG context
def extract_features_rag(text):
    prompt = (
        "You are a biomedical expert.\n"
        "Extract from the following text a JSON object with exactly these keys:\n"
        "{'drug_names':[list],'indication':[list],'sponsor':'string','nct_id':'string'}.\n"
        "Output only valid JSON.\n\n"
        f"Text:\n{text}"
    )
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
        raw = r.choices[0].message.content.strip()
        raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw
        return json.loads(raw), "gpt-3.5-turbo"
    except Exception as e:
        tqdm.write(f"⚠️ OpenAI error: {e}")
        try:
            local_out = local_llm(prompt, max_new_tokens=1000)[0]["generated_text"]
            if "```json" in local_out:
                local_out = local_out.split("```json")[-1].split("```")[0]
            return json.loads(local_out.strip()), "local-llm"
        except Exception as e2:
            tqdm.write(f"⚠️ Local LLM error: {e2}")
            return {"drug_names": [], "indication": [], "sponsor": "", "nct_id": ""}, "error"

# Main processing
def main():
    meta, index = load_meta(), load_index()
    in_scope = [r for r in meta if r.get("in_scope") == 1]
    results = []

    for rec in tqdm(in_scope, desc="Extracting features"):
        base = f"{rec.get('title','')} {rec.get('abstract','')}"
        context = base + "\n\n" + retrieve_context(base, index, meta)
        data, source = extract_features_rag(context[:8000])
        results.append({
            "pmid": rec.get("pmid",""),
            "title": rec.get("title",""),
            "in_scope": 1,
            "reason": rec.get("reason",""),
            "drug_names": data.get("drug_names", []),
            "indication": data.get("indication", []),
            "sponsor": data.get("sponsor", ""),
            "nct_id": data.get("nct_id", ""),
            "model_source": source
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ {len(results)} documents processed → {OUT}")

# Run main
if __name__ == "__main__":
    main()
