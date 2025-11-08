import asyncio, json, re
from fastmcp import Client
from openai import OpenAI
from pubmed_docs_RAG_embeddings import store_pubmed_results
from datetime import date

# Load current date for LLM context
today = date.today().strftime("%Y-%m-%d")

# Load OpenAI API
with open("APIs_dictionary.json", "r") as f:
    API = json.load(f)
openai_client = OpenAI(api_key=API["OpenAI"])

# Parse user prompt into PubMed query
async def parse_prompt(prompt: str) -> str:
    r = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": ("You are a highly skilled and expert PubMed search strategist. "
                                                    "Given a user request, output a single valid PubMed search syntax query optimized for accuracy and precision. "
                                                    "Consider that current date is " + today + " and consider only when asked.\n"
                                                    "Follow these rules:\n"
                                                    "- Always produce syntactically valid PubMed queries.\n"
                                                    "- Use parentheses for all Boolean groups (e.g., keyword logic). "
                                                    "However, if the query consists only of PMID numbers, do NOT include parentheses—list them joined by OR.\n"
                                                    "- Use complete logical structures (AND, OR, NOT) with no trailing operators.\n"
                                                    "- Include (humans[MeSH Terms]) and (english[lang]) if appropriate.\n"
                                                    "- Exclude animal-only studies with: NOT (animals[MeSH Terms] NOT humans[MeSH Terms]).\n"
                                                    "- Expand key concepts using synonyms and MeSH terms joined by OR.\n"
                                                    "- Do not include explanations or commentary. Output only the final PubMed query text."
                                                    )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200
        ,temperature=0
    )
    return r.choices[0].message.content.strip()

# Extract number of results from prompt
async def extract_num(prompt: str) -> int:
    m = re.findall(r"\b\d+\b", prompt)
    return max(1, min(int(m[0]), 10000)) if m else 10

# Main function
async def main():
    # User prompt
    prompt = input("Give prompt for PubMed/clinical trials: ")
    
    # Extract number of results from prompt
    n = await extract_num(prompt)
    print(f"\nℹ️ Requested {n} results")
    
    # Parse PubMed query into LLM
    query = await parse_prompt(prompt)
    print(f"\n🔍 PubMed query:\n{query}\n")
    
    # Call MCP tool
    async with Client("http://127.0.0.1:8000/mcp") as c:
        r = await c.call_tool("search_clinical_trials", {"keyword": query, "num_results": n})
        
    # Retrieve results
    data = getattr(r, "structured_content", None) or getattr(r, "data", None)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}

    result = data.get("result") if isinstance(data, dict) else data or []
    result = result[:n]

    if not result:
        print("⚠️ No results found.")
        return

    # Store results - PubMed RAG embeddings pipeline
    store_pubmed_results(result)

    # Display results
    #for i, d in enumerate(result, 1): print(f"\n[{i}] PMID: {d.get('pmid')}\nTitle: {d.get('title')}\n{'-'*50}")

# Run main
if __name__ == "__main__":
    asyncio.run(main())
