from pathlib import Path

from huggingface_hub import HfApi, login, upload_file, upload_folder


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HF_REPO_ID = "adamyhe/procap-atlas"

login()

api = HfApi()
api.upload_large_folder(
    repo_id=HF_REPO_ID,
    folder_path="bpnet/",
    repo_type="model",
    num_workers=4,
)

upload_folder(folder_path="configs/", repo_id=HF_REPO_ID, repo_type="model")
upload_file(
    path_or_fileobj=REPO_ROOT / "config.json",
    path_in_repo="config.json",
    repo_id=HF_REPO_ID,
    repo_type="model",
)
