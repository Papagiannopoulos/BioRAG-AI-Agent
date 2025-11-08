import os, json, time, numpy as np, faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

INDEX, META = "pubmed_index.faiss", "pubmed_meta.jsonl"

with open("APIs_dictionary.json", "r") as f:
    keys = json.load(f)
client = OpenAI(api_key=keys.get("OpenAI"))

def load_meta():
    if not os.path.exists(META):
        return []
    with open(META, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def embed_query(text):
    try:
        r = client.embeddings.create(model="text-embedding-3-small", input=text)
        v = np.array(r.data[0].embedding, dtype="float32")
    except Exception:
        v = SentenceTransformer("all-MiniLM-L6-v2").encode([text], convert_to_numpy=True)[0].astype("float32")
    return v / max(np.linalg.norm(v), 1e-9)

def semantic_search(query, top_k=20):
    if not os.path.exists(INDEX) or not os.path.exists(META):
        return []
    index = faiss.read_index(INDEX)
    meta = load_meta()
    qv = embed_query(query).reshape(1, -1)
    scores, idxs = index.search(qv, top_k)
    return [meta[i] for i in idxs[0] if 0 <= i < len(meta)]

def llm_reasoner(prompt):
    system = (
        "You are an intelligent PubMed document assistant. "
        "Given a user question, output one JSON command that best describes what to do. "
        "Available actions:\n"
        "- semantic_search: retrieve relevant documents.\n"
        "- filter_by_keyword: return docs containing a keyword.\n"
        "- count_by_topic: count docs related to a topic.\n"
        "- filter_by_year: get docs from the last N years.\n"
        "Respond ONLY with JSON in the form {\"action\": str, \"args\": {...}}."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"[Warning] Reasoner fallback due to error: {e}")
        p = prompt.lower()
        if "how many" in p or "count" in p or "number" in p:
            return {"action": "count_by_topic", "args": {"topic": p.split()[-1]}}
        if "year" in p or "recent" in p or "last" in p:
            return {"action": "filter_by_year", "args": {"years": 3}}
        if "nct" in p:
            return {"action": "filter_by_keyword", "args": {"keyword": "NCT"}}
        return {"action": "semantic_search", "args": {"query": prompt}}

def execute_action(spec):
    meta, act, args = load_meta(), spec.get("action"), spec.get("args", {})
    if act == "semantic_search":
        return semantic_search(args.get("query", ""))
    if act == "filter_by_keyword":
        kw = args.get("keyword", "").lower()
        return [m for m in meta if kw in (m["title"] + m["abstract"]).lower()]
    if act == "count_by_topic":
        kw = args.get("topic", "").lower()
        return {"count": sum(kw in (m["title"] + m["abstract"]).lower() for m in meta), "topic": kw}
    if act == "filter_by_year":
        cutoff = time.time() - int(args.get("years", 3)) * 365 * 24 * 3600
        return [m for m in meta if m.get("timestamp", 0) >= cutoff]
    return []

def summarize(query, docs):
    if not docs:
        return "No relevant studies found."
    ctx = "\n\n".join(f"Title: {d['title']}\nAbstract: {d['abstract']}" for d in docs[:10])
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize biomedical findings concisely."},
                {"role": "user", "content": f"Question: {query}\n\n{ctx}"}
            ],
            temperature=0,
            max_tokens=300
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Warning] Summarization fallback due to error: {e}")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeds = model.encode([d['abstract'] for d in docs], convert_to_numpy=True)
        idx = int(np.argmax(np.linalg.norm(embeds, axis=1)))
        return f"Representative study:\n{docs[idx]['title']}\n\n{docs[idx]['abstract'][:600]}..."

def chat_loop():
    print("PubMed Semantic Chat Ready (Ctrl+C to exit)")
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        spec = llm_reasoner(q)
        res = execute_action(spec)
        if isinstance(res, dict) and "count" in res:
            print(f"→ {res['count']} docs for topic '{res['topic']}'")
        elif not res:
            print("No results found.")
        else:
            print("\nSummary:\n", summarize(q, res))

if __name__ == "__main__":
    chat_loop()
