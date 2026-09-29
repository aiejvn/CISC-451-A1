"""Local-first Hugging Face weight resolution.

First call for a repo_id: nothing cached, so this downloads and stores it under
the standard HF cache (~/.cache/huggingface/hub, or $HF_HOME/$HF_HUB_CACHE).
Every later call finds it locally and returns immediately with no network
request at all (not even a revision check), so repeated runs never depend on
connectivity or re-download anything.
"""
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def resolve_local_path(repo_id: str) -> str:
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"[hf_cache] {repo_id} not cached yet; downloading...")
        return snapshot_download(repo_id, local_files_only=False)
