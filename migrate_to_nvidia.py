#!/usr/bin/env python3
"""
Migrate all GenAI LangChain tutorial notebooks from OpenAI/Google to NVIDIA NIM.

This script replaces:
  - ChatOpenAI       → ChatNVIDIA
  - OpenAIEmbeddings → NVIDIAEmbeddings
  - ChatGoogleGenerativeAI → ChatNVIDIA
  - ChatOllama       → ChatNVIDIA
  - All OpenAI model names → NVIDIA NIM equivalents
  - All env-var references (OPENAI_API_KEY → NVIDIA_API_KEY)

Run:  python migrate_to_nvidia.py
"""

import json
import os
import re
import glob

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model mapping ─────────────────────────────────────────────────────────────
CHAT_MODEL_MAP = {
    "gpt-4-turbo":             "meta/llama-3.3-70b-instruct",
    "gpt-4-turbo-2024-04-09":  "meta/llama-3.3-70b-instruct",
    "gpt-4o":                  "meta/llama-3.3-70b-instruct",
    "gpt-4o-mini":             "meta/llama-3.1-8b-instruct",
    "gpt-3.5-turbo":           "meta/llama-3.1-8b-instruct",
    "gpt-4":                   "meta/llama-3.3-70b-instruct",
    "gpt-3.5":                 "meta/llama-3.1-8b-instruct",
    "gpt-5-nano":              "meta/llama-3.3-70b-instruct",
    "gpt-5.4-mini":            "meta/llama-3.1-8b-instruct",
    "gemini-2.5-flash":        "meta/llama-3.3-70b-instruct",
    "gemini-2.0-flash-exp":    "meta/llama-3.3-70b-instruct",
    "qwen3:1.7b":              "meta/llama-3.1-8b-instruct",
    "deepseek-r1:latest":      "meta/llama-3.3-70b-instruct",
}

EMBED_MODEL_MAP = {
    "text-embedding-3-small":    "nvidia/llama-nemotron-embed-1b-v2",
    "text-embedding-3-large":    "nvidia/llama-nemotron-embed-1b-v2",
    "text-embedding-ada-002":    "nvidia/llama-nemotron-embed-1b-v2",
    "models/gemini-embedding-001": "nvidia/llama-nemotron-embed-1b-v2",
    "models/embedding-001":      "nvidia/llama-nemotron-embed-1b-v2",
    "nomic-embed-text":          "nvidia/llama-nemotron-embed-1b-v2",
    "embeddinggemma:latest":     "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "nvidia/llama-nemotron-embed-1b-v2",
}

# Read API key from environment — never hardcode secrets
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(NOTEBOOK_DIR, ".env"))
except ImportError:
    pass

# ── Source-level text replacements (applied to every cell source string) ──────
TEXT_REPLACEMENTS = [
    # ---- Import replacements ----
    # ChatOpenAI → ChatNVIDIA
    (
        r"from langchain_openai import ChatOpenAI",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA",
    ),
    # OpenAIEmbeddings → NVIDIAEmbeddings
    (
        r"from langchain_openai import OpenAIEmbeddings",
        "from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings",
    ),
    # Combined import: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    (
        r"from langchain_openai import ChatOpenAI,\s*OpenAIEmbeddings",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings",
    ),
    # Google Gemini → ChatNVIDIA
    (
        r"from langchain_google_genai import ChatGoogleGenerativeAI",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA",
    ),
    # Ollama → ChatNVIDIA
    (
        r"from langchain_ollama import ChatOllama",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA",
    ),
    (\
        r"from langchain_ollama import OllamaEmbeddings",
        "from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings",
    ),
    # Groq → ChatNVIDIA
    (
        r"from langchain_groq import ChatGroq",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA",
    ),
    # Generic langchain_openai import (catch-all)
    (
        r"from langchain_openai import",
        "from langchain_nvidia_ai_endpoints import",
    ),
    # Fix wrong import paths (community doesn't have ChatNVIDIA)
    (
        r"from langchain_community\.chat_models import ChatNVIDIA",
        "from langchain_nvidia_ai_endpoints import ChatNVIDIA",
    ),
    (
        r"from langchain_community\.embeddings import NVIDIAEmbeddings",
        "from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings",
    ),

    # ---- Class name replacements ----
    (r"\bChatOpenAI\b", "ChatNVIDIA"),
    (r"\bOpenAIEmbeddings\b", "NVIDIAEmbeddings"),
    (r"\bChatGoogleGenerativeAI\b", "ChatNVIDIA"),
    (r"\bChatOllama\b", "ChatNVIDIA"),
    (r"\bOllamaEmbeddings\b", "NVIDIAEmbeddings"),
    (r"\bOllamaLLM\b", "ChatNVIDIA"),
    (r"\bChatGroq\b", "ChatNVIDIA"),

    # ---- API key env-var replacements ----
    (
        r'os\.getenv\("OPENAI_API_KEY"\)',
        'os.getenv("NVIDIA_API_KEY")',
    ),
    (
        r"os\.getenv\('OPENAI_API_KEY'\)",
        "os.getenv('NVIDIA_API_KEY')",
    ),
    (
        r'os\.getenv\("GEMINI_API_KEY"\)',
        'os.getenv("NVIDIA_API_KEY")',
    ),
    # Env-var string literals in prints / checks
    (r'"OPENAI_API_KEY"', '"NVIDIA_API_KEY"'),
    (r"'OPENAI_API_KEY'", "'NVIDIA_API_KEY'"),
    (r'"GEMINI_API_KEY"', '"NVIDIA_API_KEY"'),
    (r"'GEMINI_API_KEY'", "'NVIDIA_API_KEY'"),

    # ---- pip install / package name replacements ----
    (r"langchain-openai", "langchain-nvidia-ai-endpoints"),
    (r"langchain_openai", "langchain_nvidia_ai_endpoints"),
    (r"langchain-google-genai", "langchain-nvidia-ai-endpoints"),
    (r"langchain_google_genai", "langchain_nvidia_ai_endpoints"),
    (r"langchain-ollama", "langchain-nvidia-ai-endpoints"),
    (r"langchain_ollama", "langchain_nvidia_ai_endpoints"),

    # ---- URL replacements ----
    (
        r"https://platform\.openai\.com/api-keys",
        "https://build.nvidia.com/",
    ),
    (
        r"https://platform\.openai\.com/docs",
        "https://docs.nvidia.com/nim/",
    ),
    (
        r"https://platform\.openai\.com",
        "https://build.nvidia.com/",
    ),

    # ---- Error message text ----
    (
        r"OPENAI_API_KEY not set",
        "NVIDIA_API_KEY not set",
    ),
    (
        r"OPENAI_API_KEY not found",
        "NVIDIA_API_KEY not found",
    ),
    (
        r"Add: OPENAI_API_KEY=",
        "Add: NVIDIA_API_KEY=",
    ),

    # ---- Markdown narrative replacements ----
    (r"`OPENAI_API_KEY`", "`NVIDIA_API_KEY`"),
    (r"`GEMINI_API_KEY`", "`NVIDIA_API_KEY`"),
]

# ── Model string replacements for quoted model names ─────────────────────────
def replace_model_names(text: str) -> str:
    """Replace model name strings like 'gpt-4-turbo' with NVIDIA equivalents."""
    all_models = {**CHAT_MODEL_MAP, **EMBED_MODEL_MAP}
    for old_model, new_model in all_models.items():
        # Handles both escaped-quote variants (in JSON) and plain variants
        text = text.replace(f'\\"{old_model}\\"', f'\\"{new_model}\\"')
        text = text.replace(f"\\'{old_model}\\'", f"\\'{new_model}\\'")
        text = text.replace(f'"{old_model}"', f'"{new_model}"')
        text = text.replace(f"'{old_model}'", f"'{new_model}'")
    return text


def migrate_source(source_lines: list) -> list:
    """Apply all text replacements to a cell's source lines."""
    result = []
    for line in source_lines:
        # Apply regex replacements
        for pattern, replacement in TEXT_REPLACEMENTS:
            line = re.sub(pattern, replacement, line)
        # Apply model-name replacements
        line = replace_model_names(line)
        result.append(line)
    return result


def migrate_notebook(path: str) -> int:
    """Migrate a single notebook. Returns number of cells modified."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells_modified = 0
    for cell in nb.get("cells", []):
        original = list(cell.get("source", []))
        cell["source"] = migrate_source(original)
        if cell["source"] != original:
            cells_modified += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return cells_modified


def create_env_file():
    """Create .env file only if one doesn't already exist (preserves user keys)."""
    env_path = os.path.join(NOTEBOOK_DIR, ".env")
    if os.path.exists(env_path):
        print(f"⚪ {env_path} already exists — skipping (won't overwrite your keys)")
        return
    content = """# NVIDIA NIM API
NVIDIA_API_KEY=nvapi-your-key-here

# Optional: NVIDIA model overrides
NVIDIA_CHAT_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
"""
    with open(env_path, "w") as f:
        f.write(content)
    print(f"✅ Created {env_path} — add your NVIDIA_API_KEY before running notebooks")


def update_env_example():
    """Update .env.example with NVIDIA credentials."""
    env_path = os.path.join(NOTEBOOK_DIR, ".env.example")
    content = """# NVIDIA NIM API
NVIDIA_API_KEY=nvapi-your-key-here

# Optional: NVIDIA model overrides
NVIDIA_CHAT_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
"""
    with open(env_path, "w") as f:
        f.write(content)
    print(f"✅ Updated {env_path}")


def update_requirements():
    """Replace OpenAI/Google/Ollama packages with NVIDIA in requirements.txt."""
    req_path = os.path.join(NOTEBOOK_DIR, "requirements.txt")
    if not os.path.exists(req_path):
        return

    with open(req_path, "r") as f:
        content = f.read()

    replacements = {
        "langchain-openai":        "langchain-nvidia-ai-endpoints",
        "openai":                   "# openai  (replaced by langchain-nvidia-ai-endpoints)",
        "langchain-google-genai":  "# langchain-google-genai  (replaced by langchain-nvidia-ai-endpoints)",
        "google-generativeai":     "# google-generativeai  (replaced by langchain-nvidia-ai-endpoints)",
        "langchain-huggingface":   "# langchain-huggingface  (optional — not needed for NVIDIA NIM)",
    }

    lines = content.split("\n")
    new_lines = []
    nvidia_added = False
    for line in lines:
        stripped = line.strip()
        # Skip comment lines — pass through
        if stripped.startswith("#"):
            new_lines.append(line)
            continue

        replaced = False
        for old_pkg, new_pkg in replacements.items():
            if stripped == old_pkg or stripped.startswith(f"{old_pkg}==") or stripped.startswith(f"{old_pkg}>="):
                if old_pkg == "langchain-openai" and not nvidia_added:
                    new_lines.append("langchain-nvidia-ai-endpoints")
                    nvidia_added = True
                else:
                    new_lines.append(new_pkg)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    with open(req_path, "w") as f:
        f.write("\n".join(new_lines))
    print(f"✅ Updated {req_path}")


def update_gitignore():
    """Ensure .env is in .gitignore."""
    gitignore_path = os.path.join(NOTEBOOK_DIR, ".gitignore")
    lines = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            lines = set(f.read().strip().split("\n"))

    needed = {".env", ".venv/", "__pycache__/", "*.pyc", "faiss_index_notebook/", "*.db", "*.db-shm", "*.db-wal"}
    added = needed - lines
    if added:
        with open(gitignore_path, "w") as f:
            f.write("\n".join(sorted(lines | needed)) + "\n")
        print(f"✅ Updated {gitignore_path} — added {', '.join(sorted(added))}")
    else:
        print(f"⚪ {gitignore_path} already up to date")


def main():
    print("=" * 60)
    print("  GenAI LangChain Tutorials → NVIDIA NIM Migration")
    print("=" * 60)
    print()

    # Step 1: Environment files
    print("── Step 1: Environment Files ──")
    create_env_file()
    update_env_example()
    update_gitignore()

    # Step 2: Update requirements.txt
    print("\n── Step 2: Requirements ──")
    update_requirements()

    # Step 3: Migrate all notebooks
    print("\n── Step 3: Notebook Migration ──")
    notebooks = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))
    print(f"📓 Found {len(notebooks)} notebooks\n")

    total_cells = 0
    for nb_path in notebooks:
        nb_name = os.path.basename(nb_path)
        n = migrate_notebook(nb_path)
        status = f"✅ {n} cell(s) modified" if n > 0 else "⚪ No changes needed"
        print(f"  {nb_name:55s} {status}")
        total_cells += n

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Migration complete! {total_cells} total cells updated.")
    print(f"  Provider:  NVIDIA NIM (langchain-nvidia-ai-endpoints)")
    print(f"")
    print(f"  Chat model mapping:")
    for oai, nv in CHAT_MODEL_MAP.items():
        print(f"    {oai:25s} → {nv}")
    print(f"")
    print(f"  Embedding model mapping:")
    for oai, nv in EMBED_MODEL_MAP.items():
        print(f"    {oai:25s} → {nv}")
    print("=" * 60)
    print("\n📌 Next steps:")
    print("  1. pip install langchain-nvidia-ai-endpoints")
    print("  2. Add your NVIDIA API key to .env:")
    print("       NVIDIA_API_KEY=nvapi-your-key-here")
    print("  3. Get a free key at https://build.nvidia.com/")
    print("  4. jupyter notebook")
    print("  5. Run notebooks — they now use NVIDIA NIM! 🚀")


if __name__ == "__main__":
    main()
