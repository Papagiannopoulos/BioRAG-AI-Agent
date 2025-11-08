import os, json, time, hashlib, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initiate output
INDEX = "Output/pubmed_index.faiss"
META = "Output/pubmed_meta.jsonl"
BATCH_SIZE = 5

# Load OpenAI API
with open("APIs_dictionary.json", "r") as f:
    keys = json.load(f)
client = OpenAI(api_key=keys.get("personal_OpenAI") or keys["OpenAI"])

# Hashing
def _hash(title, abstract):
    return hashlib.sha256((title + abstract).encode("utf-8")).hexdigest()

# Load existing metadata
def _load_meta():
    if not os.path.exists(META):
        return {}
    with open(META, encoding="utf-8") as f:
        return {json.loads(l)["hash"]: json.loads(l) for l in f if l.strip()}

# Save metadata
def _save_meta(items):
    with open(META, "a", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

# RAG system
def _embed(texts):
    try:
        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = np.array([x.embedding for x in r.data], dtype="float32")
    except Exception:
        local = SentenceTransformer("all-roberta-large-v1") #SentenceTransformer("all-MiniLM-L6-v2")
        vecs = local.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)

# Rule based classification
def _rule_classify(text):
    t = text.lower()
    out_words = [
        "biosimilar", "generic", "post-approval", "phase 4", "real-world", "observational",
        "retrospective", "meta-analysis", "systematic review", "preclinical", "in vitro",
        "mice", "financial", "socioeconomic", "sociodemographic", "panel", "follow-up",
        "pilot study", "case report", "case series", "review article", "literature review",
        "animal study", "non-industry sponsored"
    ]
    in_words = [
        "phase 1", "phase 2", "phase 3", "randomized", "clinical trial", "double-blind",
        "placebo-controlled", "efficacy", "safety", "industry-sponsored", "interventional"
    ]
    #if re.search(r"\bNCT\d{4,}\b", text, re.I):
    #    return 1, "NCTDetected"
    for w in out_words:
        if w in t:
            return 0, f" {w}"
    for w in in_words:
        if w in t:
            return 1, f" {w}"
    return 0, "Unclassified"

# Hybrid LLM classification
def classify_scope_llm_batch(docs):
    """Classify each PubMed abstract using rule-based + dual LLM approach."""

    def run_llm(model_name, docs):
        """Run one LLM model and return parsed JSON results."""
        prompt = (
            "Classify each PubMed abstract as in_scope (1) or out_of_scope (0).\n"
            "In-scope: phase 1–3, randomized, clinical trial, double-blind, placebo-controlled, "
            "efficacy, safety, industry-sponsored, interventional, or document related unique identifiers like NCT#####.\n"
            "Out-of-scope: biosimilar, generic, post-approval, phase 4, real-world, observational, "
            "retrospective, meta-analysis, systematic review, preclinical, in vitro, mice, "
            "financial, socioeconomic, sociodemographic, panel, follow-up, "
            "pilot study, case report, case series, review article, literature review, animal study, non-industry sponsored\n"
            "Prefer 0 if unsure.\n"
            "Return JSON: [{\"index\": i, \"in_scope\": 1 or 0, \"reason\": \"text\"}]\n"
        )
        for i, d in enumerate(docs):
            prompt += f"\n[{i+1}] Title: {d['title']}\nAbstract: {d['abstract']}\n"

        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert document classifier. Return strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=400
            )
            return json.loads(r.choices[0].message.content.strip())
        except Exception:
            return []

    # Run main LLM
    llm1 = run_llm("gpt-3.5-turbo", docs)
    llm2 = run_llm("gpt-4o-mini", docs) # or skip this and later replace with a local LLM e.g. CrossEncoder("cross-encoder/nli-roberta-base")

    results = []
    for i, d in enumerate(docs):
        rule_scope, rule_reason = _rule_classify(d["title"] + " " + d["abstract"])
        l1 = next((x for x in llm1 if x.get("index") in {i, i + 1}), None)
        l2 = next((x for x in llm2 if x.get("index") in {i, i + 1}), None)

        # Fallback defaults
        l1_scope, l2_scope = (l1 or {}).get("in_scope"), (l2 or {}).get("in_scope")
        l1_reason, l2_reason = (l1 or {}).get("reason", ""), (l2 or {}).get("reason", "")

        # If neither LLM succeeded
        if l1_scope is None and l2_scope is None:
            results.append({
                "in_scope": rule_scope,
                "reason": f"Rule:{rule_reason}"
            })
            continue

        # Normalize
        l1_scope = int(l1_scope) if l1_scope is not None else rule_scope
        l2_scope = int(l2_scope) if l2_scope is not None else rule_scope

        # Compare outcomes
        if rule_scope == l1_scope == l2_scope:
            final_scope = rule_scope
            reason = f"Matched:{rule_reason}|{l1_reason}|{l2_reason}"
        else:
            if (rule_scope == 0 and (l1_scope == 1 or l2_scope == 1)):
                final_scope = 1
                reason = f"NotMatched:{l1_reason}|{l2_reason}|Rule:{rule_reason}"
            elif (rule_scope == 1 and (l1_scope == 0 and l2_scope == 0)):
                final_scope = 0
                reason = f"NotMatched:{l1_reason}|{l2_reason}|Rule:{rule_reason}"
            else:
                final_scope = rule_scope
                reason = f"Rule:{rule_reason}"

        results.append({"in_scope": final_scope, "reason": reason})

    return results


# Store PubMed results with RAG embeddings
def store_pubmed_results(results):
    existing = _load_meta()
    new_meta, embed_texts = [], []

    for i in range(0, len(results), BATCH_SIZE):
        batch = results[i:i + BATCH_SIZE]
        docs = [{"title": r.get("title", "").strip(), "abstract": r.get("abstract", "").strip()} for r in batch]
        if not any(d["title"] or d["abstract"] for d in docs):
            continue

        cls = classify_scope_llm_batch(docs)
        time.sleep(1)

        for r, c in zip(batch, cls):
            title, abstract = r.get("title", "").strip(), r.get("abstract", "").strip()
            if not (title or abstract):
                continue
            h = _hash(title, abstract)
            if h in existing:
                continue

            rule_scope, rule_reason = _rule_classify(f"{title} {abstract}")
            in_scope = 1 if rule_scope == 1 else c["in_scope"]
            reason = c["reason"] if c["reason"].startswith("Rule:") else f"{c['reason']}|{rule_reason}"

            new_meta.append({
                "pmid": r.get("pmid", "N/A").strip(),
                "title": title,
                "abstract": abstract,
                "hash": h,
                "timestamp": time.time(),
                "in_scope": in_scope,
                "reason": reason
            })
            if in_scope:
                embed_texts.append(f"{title}\n\n{abstract}")

    if not new_meta:
        return
    if embed_texts:
        vecs = _embed(embed_texts)
        index = faiss.read_index(INDEX) if os.path.exists(INDEX) else faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, INDEX)

    _save_meta(new_meta)
