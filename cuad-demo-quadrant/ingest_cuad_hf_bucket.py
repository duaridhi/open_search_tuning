# %%
# Imports and Setup
import os
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, login
from cuad_download_utils import download_cuad_dataset, find_pdfs
from dotenv import load_dotenv

load_dotenv()
# Configuration
HF_USERNAME = "ginntonicfun"
HF_REPO_ID = f"{HF_USERNAME}/cuad-pdf-contracts"
REPO_TYPE = "dataset"
CUAD_PDF_DIR = "/home/ridhi/projects/project1/open_search_tuning/cuad_opensearch/cuad_data"

# %%
print(os.path.abspath(CUAD_PDF_DIR))

# %%
# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

print("Logging in to Hugging Face...")
login(token=hf_token)
hf_api = HfApi()
print("✓ Login successful")

# Create dataset repository if it doesn't exist
try:
    hf_api.create_repo(repo_id=HF_REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    print(f"✓ Repository '{HF_REPO_ID}' ready")
except Exception as e:
    print(f"⚠ Repository creation warning: {e}")

# %%
print("Download the CUAD dataset")
download_cuad_dataset(local_dir=CUAD_PDF_DIR, max_workers=8)

# %%
# Count and Display Files in downloaded directory
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

cuad_data_path = (Path(CUAD_PDF_DIR) / "CUAD_v1/").resolve()
pdf_list = find_pdfs(cuad_data_path)
if cuad_data_path.exists():
    pdf_count = len(pdf_list)
    print(f"✓ PDF documents found: {pdf_count}")

    if pdf_count == 510:
        print("✓ All 510 contracts downloaded successfully!")
    else:
        print(f"⚠ Expected 510 PDFs, found {pdf_count}")
else:
    print(f"✗ Data directory not found: {cuad_data_path}")
    raise FileNotFoundError(f"CUAD_v1 directory not found at {cuad_data_path}")


# %%
# Upload PDFs to Hugging Face dataset repository
print("\n" + "=" * 60)
print("UPLOADING PDFs TO HUGGING FACE")
print("=" * 60)

# Fetch existing files in the repo to support skipping already-uploaded files
print("Fetching existing files from repository...")
try:
    existing_files = set(hf_api.list_repo_files(repo_id=HF_REPO_ID, repo_type=REPO_TYPE))
    print(f"✓ Found {len(existing_files)} existing file(s) in repository")
except Exception as e:
    print(f"⚠ Could not fetch existing files (will attempt all uploads): {e}")
    existing_files = set()

new_uploads = 0
skipped = 0
failed_count = 0
failed_files = []

for pdf_file in pdf_list:
    pdf_file_path = os.path.abspath(pdf_file)
    basename = os.path.basename(pdf_file)
    # Normalize extension to lowercase .pdf
    stem, ext = os.path.splitext(basename)
    basename = stem + ext.lower()
    repo_path = "raw/" + basename

    # Skip files already in the repository
    if repo_path in existing_files:
        skipped += 1
        continue

    try:
        hf_api.upload_file(
            path_or_fileobj=pdf_file_path,
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
        )
        new_uploads += 1
        print(f"✓ Uploaded: {basename}")

    except Exception as e:
        failed_count += 1
        failed_files.append(str(pdf_file_path))
        print(f"✗ Failed: {pdf_file_path} - {e}")


# %%
# Upload Summary
print("\n" + "=" * 60)
print("UPLOAD SUMMARY")
print("=" * 60)
print(f"New files uploaded: {new_uploads}")
print(f"Files skipped (already exist): {skipped}")
print(f"Failed uploads: {failed_count}")
print(f"Total processed: {new_uploads + skipped + failed_count}")

if failed_files:
    print(f"\nFailed files:")
    for f in failed_files[:10]:
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files) - 10} more")

print("=" * 60)


# %%
# Utility Functions for Hugging Face Operations

def file_exists_in_hf(file_name: str, repo_id: str = HF_REPO_ID) -> bool:
    """
    Check if a file exists in a Hugging Face dataset repository.

    Parameters:
        file_name (str): Path of the file in the repo (e.g., 'raw/example.pdf')
        repo_id (str): Hugging Face repository ID

    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        files = set(hf_api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE))
        return file_name in files
    except Exception:
        return False


def get_file_from_hf(file_name: str, repo_id: str = HF_REPO_ID) -> bytes | None:
    """
    Retrieve a file from a Hugging Face dataset repository.

    Parameters:
        file_name (str): Path of the file in the repo (e.g., 'raw/example.pdf')
        repo_id (str): Hugging Face repository ID

    Returns:
        bytes: Binary content of the file, or None if file not found
    """
    try:
        if not file_exists_in_hf(file_name, repo_id):
            print(f"✗ File not found: {file_name}")
            return None

        local_path = hf_api.hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type=REPO_TYPE,
        )
        with open(local_path, "rb") as f:
            file_content = f.read()
        print(f"✓ Retrieved {file_name} from repository '{repo_id}'")
        return file_content

    except Exception as e:
        print(f"✗ Error retrieving {file_name}: {e}")
        raise

# %%
