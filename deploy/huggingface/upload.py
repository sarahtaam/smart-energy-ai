"""Deploy the live demo to a Hugging Face Space (Docker SDK), creating the Space if needed.

    pip install huggingface_hub
    hf auth login                                   # once, with a token that has *write* permission
    python deploy/huggingface/upload.py <hf-user>/smart-energy-ai [--github-url https://github.com/...]

The token can also come from the HF_TOKEN environment variable (that is how the CI uses it).
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE_DIRS = ["smart_energy", "dashboard", "deploy"]


def assemble(dest: Path, github_url: str) -> None:
    """Space layout = the code + the demo Dockerfile and the Space README at its root."""
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ["requirements.txt", ".dockerignore"]:
        shutil.copy(ROOT / f, dest / f)
    shutil.copy(HERE / "Dockerfile", dest / "Dockerfile")
    readme = (HERE / "README.md").read_text(encoding="utf-8").replace("GITHUB_REPO_URL", github_url)
    (dest / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    # bash refuses CRLF scripts, which a Windows checkout can produce
    script = dest / "deploy/huggingface/start.sh"
    script.write_bytes(script.read_bytes().replace(b"\r\n", b"\n"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="<hf-user>/<space-name>")
    ap.add_argument("--github-url", default="see the project's GitHub repository",
                    help="link shown on the Space page")
    ap.add_argument("--private", action="store_true", help="create the Space as private (first deploy only)")
    args = ap.parse_args()

    api = HfApi()
    url = api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        assemble(Path(tmp), args.github_url)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy Smart Energy AI demo",
                          delete_patterns=[f"{d}/**" for d in CODE_DIRS])  # drop files removed locally
    owner, name = args.space.split("/")
    print(f"Uploaded to {url}\nBuild logs: {url} (tab 'Logs'), ready in ~5 min")
    print(f"Demo URL:   https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.hf.space")


if __name__ == "__main__":
    main()
