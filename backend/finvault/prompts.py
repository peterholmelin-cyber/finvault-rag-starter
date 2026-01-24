from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text().strip()


def get_prompt(name: str, **kwargs) -> str:
    template = load_prompt(name)
    if kwargs:
        return template.format(**kwargs)
    return template


# Prompt names as constants for IDE autocompletion
CONTEXT_ENRICHMENT = "context_enrichment"
VLM_IMAGE_DESCRIPTION = "vlm_image_description"
HYDE_EXPANSION = "hyde_expansion"
MULTI_QUERY_EXPANSION = "multi_query_expansion"
RAG_SYSTEM = "rag_system"
RAG_QUERY = "rag_query"
