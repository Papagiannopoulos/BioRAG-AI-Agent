import requests
import json
import time
from xml.etree import ElementTree as ET
from fastmcp import FastMCP

# Load APIs
with open("APIs_dictionary.json", "r") as f:
    PUBMED_API_KEY = json.load(f).get("Pubmed")
# Initialize MCP
mcp = FastMCP(
    name="PubMed_Search_MCP",
    instructions="MCP tool for retrieving PubMed clinical trial metadata."
)

@mcp.tool()
def search_clinical_trials(keyword: str, num_results: int = 10) -> dict:
    base_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    try:
        # Base search - pmids
        r = requests.get(base_esearch, params={
            "db": "pubmed",
            "term": keyword,
            "retmode": "json",
            "retmax": min(num_results, 10000),
            "api_key": PUBMED_API_KEY,
        }, timeout=30)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"result": [], "message": "No articles found."}
        
        # Fetch details - title/abstract
        out = []
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            f = requests.get(base_efetch, params={
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "api_key": PUBMED_API_KEY,
            }, timeout=30)
            f.raise_for_status()
            try:
                root = ET.fromstring(f.text)
            except ET.ParseError:
                continue
            
            # Parse articles
            for art in root.findall(".//PubmedArticle"):
                pmid = art.findtext(".//PMID") or "N/A"
                title = art.findtext(".//ArticleTitle") or "N/A"

                parts = []
                for node in art.findall(".//AbstractText"):
                    text = (node.text or "").strip()
                    if not text:
                        continue
                    label = node.attrib.get("Label")
                    category = node.attrib.get("NlmCategory")
                    prefix = ""
                    if label and category:
                        prefix = f"{label} ({category}): "
                    elif label:
                        prefix = f"{label}: "
                    elif category:
                        prefix = f"{category}: "
                    parts.append(prefix + text)

                for node in art.findall(".//OtherAbstract/AbstractText"):
                    text = (node.text or "").strip()
                    if text and text not in parts:
                        parts.append(text)

                if not parts:
                    text = art.findtext(".//Abstract")
                    if text:
                        parts.append(text.strip())

                abstract = " ".join(parts) if parts else "N/A"

                out.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                })
            
            # Respect API call limits
            time.sleep(0.34)

        return {"result": out[:num_results], "message": f"Returned {len(out[:num_results])} records."}

    except Exception as e:
        return {"result": [], "error": str(e)}

# run MCP server
if __name__ == "__main__":
    print("MCP Server active at http://127.0.0.1:8000/mcp")
    mcp.run(transport="http", host="127.0.0.1", port=8000, log_level="INFO")
