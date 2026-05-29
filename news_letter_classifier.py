import os
import re
import math
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CLASSIFY_MODEL = os.getenv("OLLAMA_CLASSIFY_MODEL", "mistral")
OLLAMA_EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# nomic shortlisting — topics below this score are excluded before sending to Mistral
MULTI_LOW_THRESHOLD   = float(os.getenv("MULTI_LOW_THRESHOLD", "0.35"))
# how many top topics to send to Mistral for final decision
TOP_K_FOR_MISTRAL     = int(os.getenv("TOP_K_FOR_MISTRAL", "6"))
MAX_TOPICS            = int(os.getenv("MAX_TOPICS", "5"))

NEWSLETTER_TOPICS = [
    "ai",
    "machine_learning",
    "dev_tools",
    "open_source",
    "cybersecurity",
    "cloud_infrastructure",
    "web_development",
    "data_science",
    "startups_funding",
    "research",
    "finance",
    "crypto",
    "fintech",
    "health",
    "sports",
    "science",
    "space",
    "general_tech",
]

# Descriptions are keyword-dense and kept DISTINCT from each other intentionally.
# nomic-embed-text uses these to rank topics — the more specific the better.
_TOPIC_DESCRIPTIONS = {
    "ai": (
        "Artificial intelligence industry news: OpenAI ChatGPT, Anthropic Claude, Google Gemini, "
        "AI agents, LLM product releases, AI startups, AI regulation, AI ethics, generative AI applications, "
        "AI in business, AI tools for consumers, what AI products are doing in the world."
    ),
    "machine_learning": (
        "Technical machine learning content: model training procedures, neural network architectures, "
        "transformer internals, loss functions, gradient descent, fine-tuning techniques, LoRA, RLHF, "
        "model benchmarks, evaluation metrics, paper reviews, ML engineering, dataset curation."
    ),
    "dev_tools": (
        "Software developer productivity tools: VS Code, JetBrains, GitHub Copilot, CI/CD pipelines, "
        "debugging tools, compilers, package managers, linters, testing frameworks, terminal tools, "
        "developer experience improvements, SDKs, CLI tools, build systems."
    ),
    "open_source": (
        "Open source software ecosystem: GitHub trending repositories, new OSS project releases, "
        "open-source licenses, community contributions, pull requests, OSS funding, "
        "Linux kernel updates, Apache foundation, CNCF projects, open-source governance."
    ),
    "cybersecurity": (
        "Cybersecurity threats and defences: data breaches, ransomware attacks, zero-day exploits, "
        "CVE vulnerabilities, phishing campaigns, malware, social engineering, penetration testing, "
        "encryption standards, privacy regulations, GDPR, incident response, threat intelligence."
    ),
    "cloud_infrastructure": (
        "Cloud platforms and infrastructure: AWS, Microsoft Azure, Google Cloud Platform, "
        "Kubernetes orchestration, Docker containers, serverless functions, Terraform, "
        "infrastructure as code, DevOps practices, site reliability engineering, microservices."
    ),
    "web_development": (
        "Web development technologies: React, Next.js, Vue, Svelte, Angular, TypeScript, "
        "REST APIs, GraphQL, CSS frameworks, HTML standards, browser compatibility, "
        "web performance, frontend architecture, backend Node.js, web accessibility."
    ),
    "data_science": (
        "Data science and analytics: Pandas, NumPy, Spark, SQL, data pipelines, ETL workflows, "
        "Jupyter notebooks, data visualisation, Power BI, Tableau, feature engineering, "
        "statistical analysis, A/B testing, data warehouses, dbt, Airflow."
    ),
    "startups_funding": (
        "Startup ecosystem news: Series A/B/C venture capital funding rounds, angel investment, "
        "company acquisitions, mergers, IPOs, unicorn valuations, Y Combinator, accelerators, "
        "startup launches, founder stories, product-market fit, exit events."
    ),
    "research": (
        "Academic and scientific research: peer-reviewed journal papers, arXiv preprints, "
        "university research grants, PhD studies, research institutions, scientific methodology, "
        "conference proceedings NeurIPS ICML CVPR, literature reviews, reproducibility."
    ),
    "finance": (
        "Traditional financial markets and economy: stock prices, S&P 500 Nasdaq Dow Jones, "
        "bond yields, interest rates, Federal Reserve monetary policy, inflation CPI, "
        "GDP growth, earnings reports, ETFs, mutual funds, personal finance, banking sector, "
        "hedge funds, options trading — NOT cryptocurrency."
    ),
    "crypto": (
        "Cryptocurrency and blockchain: Bitcoin BTC price, Ethereum ETH, altcoins, "
        "DeFi decentralised finance, NFTs, Web3, crypto exchanges Coinbase Binance, "
        "blockchain protocols, token launches, crypto regulation, stablecoins, mining, wallets."
    ),
    "fintech": (
        "Financial technology companies and products: digital payment systems, Stripe PayPal, "
        "neobanks Revolut Monzo, digital wallets Apple Pay, buy-now-pay-later Klarna, "
        "lending platforms, insurtech, open banking APIs, embedded finance, regtech compliance tools."
    ),
    "health": (
        "Health, medicine and wellness: clinical trials, drug approvals FDA, disease outbreaks, "
        "medical research, fitness, nutrition, mental health, sleep science, "
        "hospital systems, pharmaceutical companies, public health policy, longevity."
    ),
    "sports": (
        "Sports events and results: cricket IPL Test matches, football Premier League La Liga, "
        "basketball NBA, tennis Grand Slam, Formula 1 race results, Olympics, "
        "player transfers, team standings, sports analytics, coaching changes."
    ),
    "science": (
        "Natural and physical sciences: physics discoveries, chemistry breakthroughs, "
        "biology genetics DNA CRISPR, climate science, quantum mechanics, materials science, "
        "neuroscience, paleontology, mathematics — NOT space exploration specifically."
    ),
    "space": (
        "Space exploration and astronomy: NASA missions, SpaceX Falcon rocket launches, "
        "ISS International Space Station, satellite constellations Starlink, Mars exploration, "
        "James Webb Space Telescope discoveries, exoplanets, black holes, asteroid missions."
    ),
    "general_tech": (
        "General technology news that spans multiple unrelated areas with no single dominant topic."
    ),
}

_topic_embeddings: dict[str, list[float]] = {}


def _clean_body(body: str, max_chars: int = 1200) -> str:
    if not body:
        return ""
    if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
        soup = BeautifulSoup(body, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = soup.get_text(separator=" ")
    body = re.sub(r"https?://\S+", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body[:max_chars]


def _embed(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_topic_embeddings() -> dict[str, list[float]]:
    global _topic_embeddings
    if not _topic_embeddings:
        for topic, desc in _TOPIC_DESCRIPTIONS.items():
            _topic_embeddings[topic] = _embed(desc)
    return _topic_embeddings


def _shortlist_topics(subject: str, body: str) -> list[str]:
    """
    Use nomic-embed-text to rank all topics by semantic similarity.
    Returns the top TOP_K_FOR_MISTRAL topics above the minimum threshold.
    nomic does RANKING only — Mistral does the final classification.
    """
    text = f"{subject}. {body[:900]}"
    newsletter_emb = _embed(text)
    topic_embs     = _get_topic_embeddings()

    scores = {
        topic: _cosine_similarity(newsletter_emb, emb)
        for topic, emb in topic_embs.items()
    }

    # Filter out clearly irrelevant topics, then take top K
    filtered = {t: s for t, s in scores.items() if s >= MULTI_LOW_THRESHOLD}

    if not filtered:
        # Nothing passed the floor — take top 3 regardless
        filtered = scores

    shortlist = sorted(filtered, key=filtered.__getitem__, reverse=True)[:TOP_K_FOR_MISTRAL]
    return shortlist


def _verify_with_mistral(email_data: dict, shortlist: list[str]) -> list[str]:
    """
    Mistral receives only the shortlisted topics and decides which ones
    are substantially covered by the newsletter. Always called — it is the
    final and only decision maker.
    """
    clean   = _clean_body(email_data.get("body", ""))
    subject = email_data.get("subject", "")

    topic_defs = {
        "ai":                   "AI products, companies, tools, ChatGPT, agents — the AI industry",
        "machine_learning":     "HOW models work technically — training, architectures, fine-tuning, benchmarks",
        "dev_tools":            "developer productivity tools — IDEs, CI/CD, debuggers, CLIs",
        "open_source":          "open-source project releases, GitHub trends, OSS community",
        "cybersecurity":        "security threats, breaches, malware, CVEs, privacy",
        "cloud_infrastructure": "AWS/Azure/GCP, Kubernetes, Docker, DevOps",
        "web_development":      "React, APIs, CSS, HTML, frontend/backend frameworks",
        "data_science":         "data pipelines, analytics, Pandas, SQL, BI tools",
        "startups_funding":     "VC funding rounds, acquisitions, startup launches",
        "research":             "academic papers, university studies, peer-reviewed research",
        "finance":              "stock markets, economy, Fed, inflation, bonds, ETFs — NOT crypto",
        "crypto":               "Bitcoin, Ethereum, DeFi, NFTs, blockchain, Web3",
        "fintech":              "payment apps, neobanks, digital wallets, embedded finance",
        "health":               "medicine, wellness, clinical trials, fitness, nutrition",
        "sports":               "match results, player news, league standings",
        "science":              "physics, biology, chemistry, climate — NOT space",
        "space":                "NASA, SpaceX, rockets, telescopes, astronomy",
        "general_tech":         "mixed tech with no dominant single topic",
    }

    lines = "\n".join(
        f"  - {t}: {topic_defs[t]}" for t in shortlist if t in topic_defs
    )

    prompt = f"""You are a precise multi-label newsletter classifier.

The newsletter below may cover some of these topics:
{lines}

TASK: List ONLY the topics that are SUBSTANTIALLY covered.
A topic qualifies if it occupies a meaningful portion of content — not a single sentence or passing mention.
A newsletter can have 1, 2, 3 or more qualifying topics.

DISAMBIGUATION:
- finance vs crypto: finance = stocks/bonds/economy; crypto = Bitcoin/Ethereum/blockchain specifically
- ai vs machine_learning: ai = AI industry news/products; machine_learning = technical training details
- science vs space: science = physics/biology/chemistry; space = rockets/NASA/astronomy

EXAMPLES:

Newsletter: Bitcoin ETF approved by SEC. Fed holds interest rates at 5.25%. Stripe launches crypto payment API. DeFi protocols hit record TVL.
Offered topics: finance, crypto, fintech, ai
Answer: finance, crypto, fintech

Newsletter: OpenAI releases GPT-5. Meta open-sources Llama 3. NVIDIA unveils H200 GPU. AI coding agents now write full PRs autonomously.
Offered topics: ai, machine_learning, open_source, dev_tools
Answer: ai, open_source

Newsletter: New LoRA fine-tuning paper beats full fine-tuning on benchmarks. Researchers propose new attention mechanism reducing FLOPS by 40%.
Offered topics: ai, machine_learning, research, data_science
Answer: machine_learning, research

Newsletter: SpaceX Starship completes orbital test. James Webb finds carbon dioxide on K2-18b. NASA announces Artemis III crew.
Offered topics: space, science, research, startups_funding
Answer: space

Newsletter: IPL 2025 final results. Manchester City wins Champions League. Formula 1 Monaco Grand Prix recap.
Offered topics: sports, finance, health
Answer: sports

Now classify:
Subject: {subject}
Body: {clean[:900]}

Reply with ONLY one line in this exact format (comma-separated, no explanation):
TOPICS: topic1, topic2, ...
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_CLASSIFY_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "top_p": 1},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["response"].strip().lower()

        match = re.search(r"topics:\s*(.+)", raw)
        if not match:
            # Fallback: try parsing the whole response as a comma-separated list
            raw_topics = [t.strip().rstrip(".,;:") for t in raw.split(",")]
        else:
            raw_topics = [t.strip().rstrip(".,;:") for t in match.group(1).split(",")]

        verified = [t for t in raw_topics if t in NEWSLETTER_TOPICS]
        return verified[:MAX_TOPICS] if verified else ["general_tech"]

    except requests.exceptions.RequestException as e:
        print(f"[newsletter_classifier] Mistral request failed: {e}")
        return ["general_tech"]


def classify_newsletter_topic(email_data: dict) -> list[str]:
    """
    Two-stage multi-label newsletter topic classifier.

    Stage 1 — nomic-embed-text:
        Scores all 18 topics by semantic similarity.
        Returns top TOP_K_FOR_MISTRAL topics as a shortlist (RANKING only).

    Stage 2 — Mistral (always called):
        Receives only the shortlisted topics.
        Decides which ones are substantially covered.
        Returns a list of verified topic labels.
    """
    subject = email_data.get("subject", "")
    body    = _clean_body(email_data.get("body", ""))

    try:
        shortlist = _shortlist_topics(subject, body)
    except Exception as e:
        print(f"[newsletter_classifier] nomic embedding failed: {e} — sending all topics to Mistral")
        shortlist = NEWSLETTER_TOPICS[:]

    return _verify_with_mistral(email_data, shortlist)
