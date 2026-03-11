# S3/MinIO Integration for Presigned URLs

## Overview

The CUAD Qdrant API now supports presigned URLs for direct PDF downloads from S3/MinIO storage. This enables the frontend to download PDFs without requiring authentication or additional API calls.

## Setup

### 1. Environment Variables

Configure the following environment variables in your `.env` file:

```bash
# MinIO/S3 Endpoint
MINIO_ENDPOINT=http://localhost:9000             # Internal endpoint (API calls)
MINIO_PUBLIC_ENDPOINT=http://localhost:9000      # Public endpoint (browser access)

# MinIO Credentials
MINIO_ACCESS_KEY=minioadmin                      # S3 access key ID
MINIO_SECRET_KEY=minioadmin                      # S3 secret access key

# Bucket Configuration
MINIO_BUCKET=cuad-contracts                      # S3 bucket name
PRESIGNED_EXPIRY_SECONDS=3600                    # URL expiry (default: 1 hour)
```

### 2. PDF Storage Location

Upload PDF contracts to S3 with this key structure:

```
s3://cuad-contracts/raw/{contract_name}.pdf
```

Example:
- `s3://cuad-contracts/raw/Acquisition Agreement.pdf`
- `s3://cuad-contracts/raw/Employment Contract.pdf`

### 3. Requirements

Ensure `boto3` is installed:

```bash
pip install -r requirements.txt
```

## API Endpoints

### @app.get("/search") - Search with PDF URLs

**Response includes `pdf_url` field:**

```json
{
  "query": "termination clauses",
  "top_k": 10,
  "strategy": "semantic_search",
  "results_count": 2,
  "results": [
    {
      "id": "chunk_123",
      "score": 0.92,
      "title": "Acquisition Agreement",
      "text": "...",
      "page_start": 5,
      "page_end": 5,
      "char_start": 1234,
      "char_end": 1567,
      "pdf_path": "raw/Acquisition Agreement.pdf",
      "pdf_url": "http://localhost:9000/cuad-contracts/raw/...?X-Amz-Algorithm=...",
      "source": ["embeddings"]
    }
  ]
}
```

### @app.get("/documents") - List Documents with URLs

**Response includes S3 metadata and presigned URLs:**

```json
{
  "total": 2,
  "documents": [
    {
      "title": "Acquisition Agreement",
      "pdf_path": "raw/Acquisition Agreement.pdf",
      "s3_key": "raw/Acquisition Agreement.pdf",
      "pdf_url": "http://localhost:9000/cuad-contracts/raw/...?X-Amz-Algorithm=...",
      "chunk_count": 145,
      "total_chars": 45231
    },
    {
      "title": "Employment Contract",
      "pdf_path": "raw/Employment Contract.pdf",
      "s3_key": "raw/Employment Contract.pdf",
      "pdf_url": "http://localhost:9000/cuad-contracts/raw/...?X-Amz-Algorithm=...",
      "chunk_count": 89,
      "total_chars": 28934
    }
  ]
}
```

## Features

- **Automatic Presigned URL Generation**: URLs are generated on-the-fly with configurable expiry
- **S3/MinIO Compatible**: Works with AWS S3, MinIO, or compatible object storage
- **URL Signature v4**: Uses AWS Signature Version 4 for compatibility
- **Graceful Degradation**: API works without S3 if storage is unavailable
- **Dual-Client Architecture**: Separate internal and public endpoints for flexibility

## Presigned URL Lifetime

By default, presigned URLs expire after **3600 seconds (1 hour)**.

To change this, set:
```bash
PRESIGNED_EXPIRY_SECONDS=7200  # 2 hours
```

## Troubleshooting

### URLs Not Generated

1. Check S3 connection:
```bash
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
```

2. Verify bucket exists:
```bash
mc ls minio/cuad-contracts/raw/
```

3. Check app logs for S3 initialization:
```
[INFO] S3 clients initialized for presigned URL generation
```

### URLs Return 404

Verify PDF path format:
- S3 key should be: `raw/{contract_name}.pdf`
- Contract name in Qdrant should match S3 file name
- Test with MinIO console: `http://localhost:9000/`

## Frontend Usage

**Download PDF from presigned URL:**

```javascript
// Direct download
const link = document.createElement('a');
link.href = searchResult.pdf_url;
link.download = `${searchResult.title}.pdf`;
link.click();

// Or embed in iframe
<iframe src={searchResult.pdf_url} />

// Or use fetch to get buffer
const response = await fetch(searchResult.pdf_url);
const blob = await response.blob();
```

## Implementation Details

See [s3_utils.py](s3_utils.py) for:
- `init_s3_clients()` - Initialize S3 clients
- `generate_presigned_url(s3_key)` - Generate URL for object
- `list_s3_documents(prefix)` - List and generate URLs for all documents
