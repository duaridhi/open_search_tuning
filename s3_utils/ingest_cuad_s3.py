# %%
# Imports and Setup
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add s3_utils directory to path for imports
s3_utils_path = Path("/home/ridhi/projects/project1/open_search_tuning/s3_utils")
if str(s3_utils_path) not in sys.path:
    sys.path.insert(0, str(s3_utils_path))

from connect_s3 import get_storage_client

# Configuration
BUCKET = "cuad-contracts"
REPO_ID = "theatticusproject/cuad"
LOCAL_DIR = "./cuad_data"




# %%
# Initialize MinIO Client
print("Connecting to MinIO...")
minio_client = get_storage_client(local=True)
print("✓ Connection successful")

# Create bucket if it doesn't exist
try:
    minio_client.create_bucket(Bucket=BUCKET)
    print(f"✓ Bucket '{BUCKET}' created")
except Exception as e:
    if "already owns the bucket" in str(e) or "BucketAlreadyExists" in str(e):
        print(f"✓ Bucket '{BUCKET}' already exists")
    else:
        print(f"⚠ Bucket creation error: {e}")


# %%
# Download CUAD Dataset from HuggingFace
print("\n" + "=" * 60)
print("DOWNLOADING CUAD DATASET FROM HUGGINGFACE to ", LOCAL_DIR)
print("=" * 60)

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=["*.pdf", "*.PDF"],  # Handle both lowercase and uppercase
        max_workers=8
    )
    print("✓ Dataset downloaded successfully to ", LOCAL_DIR)
    
except Exception as e:
    print(f"✗ Failed to download dataset: {e}")
    raise


# %%
# Count and Display Downloaded Files
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

cuad_data_path = Path(LOCAL_DIR) / "CUAD_v1"

if cuad_data_path.exists():
    pdf_count = len(find_pdfs(cuad_data_path))
    print(f"✓ PDF documents found: {pdf_count}")
    
    if pdf_count == 510:
        print("✓ All 510 contracts downloaded successfully!")
    else:
        print(f"⚠ Expected 510 PDFs, found {pdf_count}")
else:
    print(f"✗ Data directory not found: {cuad_data_path}")
    raise FileNotFoundError(f"CUAD_v1 directory not found at {cuad_data_path}")


# %%
# Upload PDFs to MinIO
print("\n" + "=" * 60)
print("UPLOADING PDFs TO MINIO")
print("=" * 60)

new_uploads = 0
skipped = 0
failed_count = 0
failed_files = []

for pdf_file in find_pdfs(cuad_data_path):
    relative_path = pdf_file.relative_to(cuad_data_path)
    s3_key = f"raw/{relative_path}"
    
    # Check if file exists in MinIO
    file_exists = False
    try:
        minio_client.head_object(Bucket=BUCKET, Key=s3_key)
        file_exists = True
    except:
        file_exists = False
    
    # Skip existing files
    if file_exists:
        skipped += 1
        continue
    
    # Upload new file
    try:
        with open(pdf_file, "rb") as f:
            minio_client.put_object(
                Bucket=BUCKET,
                Key=s3_key,
                Body=f,
                ContentType="application/pdf"
            )
        new_uploads += 1
        print(f"✓ Uploaded: {relative_path}")
    
    except Exception as e:
        failed_count += 1
        failed_files.append(str(relative_path))
        print(f"✗ Failed: {relative_path} - {e}")


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
# Utility Functions for MinIO Operations

def file_exists_in_minio(file_name, bucket_name):
    """
    Check if a file exists in MinIO bucket.
    
    Parameters:
        file_name (str): Name or path of the file (e.g., 'raw/Part_I/example.pdf')
        bucket_name (str): Name of the MinIO bucket
    
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
       # minio_client.head_object(Bucket=bucket_name, Key=file_name)
        minio_client.list_objects_v2(Bucket=bucket_name, Prefix="raw/")
        return True
    except:
        return False


def get_file_from_minio(file_name, bucket_name):
    """
    Retrieve a file from MinIO bucket.
    
    Parameters:
        file_name (str): Name or path of the file (e.g., 'raw/Part_I/example.pdf')
        bucket_name (str): Name of the MinIO bucket
    
    Returns:
        bytes: Binary content of the file, or None if file not found
    
    Raises:
        Exception: If connection or retrieval fails
    """
    try:
        if not file_exists_in_minio(file_name, bucket_name):
            print(f"✗ File not found: {file_name}")
            return None
        
        response = minio_client.get_object(Bucket=bucket_name, Key=file_name)
        file_content = response['Body'].read()
        print(f"✓ Retrieved {file_name} from bucket '{bucket_name}'")
        return file_content
    
    except Exception as e:
        print(f"✗ Error retrieving {file_name}: {e}")
        raise
# %%
minio_client.get_object(Bucket=BUCKET, Key="/full_contract_pdf/Part_I/Affiliate_Agreements/SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement.pdf")

#file_exists_in_minio(bucket_name=BUCKET,file_name="cuad-contracts/raw/full_contract_pdf/Part_I/Affiliate_Agreements/SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement.pdf")
#minio_client.list_objects_v2(Bucket=BUCKET, Prefix="raw/")


# %%
