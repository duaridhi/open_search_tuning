###############################################################################
# Core
###############################################################################

variable "aws_region" {
  description = "AWS region for all resources. Plan pins us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name, used to prefix/name resources."
  type        = string
  default     = "cuad-ai-demo"
}

variable "environment" {
  description = "Environment tag (demo/dev/prod)."
  type        = string
  default     = "demo"
}

###############################################################################
# ECR / App Runner image
###############################################################################

variable "ecr_repo_name" {
  description = "Name of the ECR repository that holds the backend image."
  type        = string
  default     = "cuad-ai-demo"
}

variable "apprunner_image_tag" {
  description = <<-EOT
    Image tag App Runner deploys from ECR. You must push an image to the ECR repo
    with this tag BEFORE App Runner can create the service successfully.
    See README.md "Apply order".
  EOT
  type        = string
  default     = "latest"
}

variable "apprunner_cpu" {
  description = "App Runner instance CPU. Plan: 0.5 vCPU."
  type        = string
  default     = "0.25 vCPU"
}

variable "apprunner_memory" {
  description = "App Runner instance memory. Plan: 1 GB."
  type        = string
  default     = "1 GB"
}

variable "apprunner_port" {
  description = "Container port App Runner routes traffic to."
  type        = number
  default     = 8080
}

variable "apprunner_auto_deploy" {
  description = "Auto-deploy on new image push to ECR (the CI/CD trigger in the plan)."
  type        = bool
  default     = true
}

# NOTE on App Runner instance sizing:
# The plan calls for 0.5 vCPU / 1 GB. App Runner only accepts a fixed set of
# (cpu, memory) pairs and 0.5 vCPU is NOT a valid value — the smallest is
# "0.25 vCPU" and the next is "1 vCPU". We default to "0.25 vCPU" / "1 GB"
# (cheapest, scales-to-zero friendly). Override apprunner_cpu to "1 vCPU" if
# you hit CPU pressure on Qdrant client init at cold start.

###############################################################################
# SageMaker Serverless endpoints (embedder + reranker)
###############################################################################

variable "hf_dlc_image_uri" {
  description = <<-EOT
    HuggingFace PyTorch Inference Deep Learning Container (CPU) image URI for
    SageMaker in us-east-1. Confirmed current tag as of 2026-06-10 from
    https://huggingface.co/docs/sagemaker/en/dlcs/available .
    If you change region, update the account-id/region in this URI accordingly
    (account 763104351884 is the AWS DLC account for us-east-1).
  EOT
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.51.3-cpu-py312-ubuntu22.04"
}

variable "embedder_hf_model_id" {
  description = "HF Hub model id pulled at SageMaker model creation for the embedder."
  type        = string
  default     = "sentence-transformers/all-MiniLM-L6-v2"
}

variable "embedder_hf_task" {
  description = "HF inference toolkit task for the embedder DLC (HF_TASK env)."
  type        = string
  default     = "feature-extraction"
}

variable "reranker_hf_model_id" {
  description = "HF Hub model id pulled at SageMaker model creation for the reranker."
  type        = string
  default     = "BAAI/bge-reranker-v2-m3"
}

variable "reranker_hf_task" {
  description = "HF inference toolkit task for the reranker DLC (HF_TASK env)."
  type        = string
  default     = "text-classification"
}

variable "embedder_endpoint_name" {
  description = "SageMaker serverless endpoint name for the embedder."
  type        = string
  default     = "cuad-embedder-serverless"
}

variable "reranker_endpoint_name" {
  description = "SageMaker serverless endpoint name for the reranker."
  type        = string
  default     = "cuad-reranker-serverless"
}

variable "serverless_memory_mb" {
  description = "SageMaker Serverless MemorySizeInMB. 2048 fits MiniLM + bge-reranker."
  type        = number
  default     = 2048
}

variable "serverless_max_concurrency" {
  description = "SageMaker Serverless MaxConcurrency."
  type        = number
  default     = 5
}

###############################################################################
# Backend (App Runner) runtime config — plain env vars (NOT secrets)
###############################################################################

variable "cluster_url" {
  description = "Qdrant Cloud URL (CLUSTER_URL). Not a secret. Set to your cluster."
  type        = string
  default     = "https://REPLACE-ME.us-east-1-0.aws.cloud.qdrant.io:6333"
}

variable "qdrant_collection" {
  description = "Qdrant collection name."
  type        = string
  default     = "cuad_contracts"
}

variable "embed_model" {
  description = "EMBED_MODEL env var (must match embedder endpoint model output dim)."
  type        = string
  default     = "sentence-transformers/all-MiniLM-L6-v2"
}

variable "vector_size" {
  description = "VECTOR_SIZE env var. 384 for MiniLM, 1024 for bge-large."
  type        = number
  default     = 384
}

variable "rerank_model" {
  description = "RERANK_MODEL env var."
  type        = string
  default     = "BAAI/bge-reranker-v2-m3"
}

variable "chat_model" {
  description = "CHAT_MODEL env var (HF Inference API LLM)."
  type        = string
  default     = "Qwen/Qwen3-235B-A22B:novita"
}

###############################################################################
# Secrets supplied at apply time (written into SSM SecureString)
###############################################################################

variable "hf_token" {
  description = <<-EOT
    HF_TOKEN value written into SSM SecureString /cuad/hf-token.
    Leave empty ("") to have Terraform create the parameter with a placeholder
    you must overwrite out-of-band, OR pass via TF_VAR_hf_token at apply time.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "qdrant_api_key" {
  description = <<-EOT
    QDRANT_API_KEY value written into SSM SecureString /cuad/qdrant-api-key.
    Same handling as hf_token.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "hf_token_param_name" {
  description = "SSM parameter name for HF_TOKEN."
  type        = string
  default     = "/cuad/hf-token"
}

variable "qdrant_api_key_param_name" {
  description = "SSM parameter name for QDRANT_API_KEY."
  type        = string
  default     = "/cuad/qdrant-api-key"
}

###############################################################################
# Frontend (S3 + CloudFront)
###############################################################################

variable "frontend_bucket_name" {
  description = "S3 bucket name for the SPA. Must be globally unique."
  type        = string
  default     = "cuad-ai-demo-fe"
}

variable "cloudfront_price_class" {
  description = "CloudFront price class. PriceClass_100 = US/EU only (cheapest)."
  type        = string
  default     = "PriceClass_100"
}

###############################################################################
# GitHub Actions OIDC
###############################################################################

variable "create_github_oidc_provider" {
  description = <<-EOT
    Create the GitHub OIDC provider in this account. Set false if the provider
    (token.actions.githubusercontent.com) already exists in the account.
  EOT
  type        = bool
  default     = true
}

variable "github_backend_repo" {
  description = "GitHub org/repo for the backend (ECR push + App Runner deploy)."
  type        = string
  default     = "RidhiD/cuad-ai-demo"
}

variable "github_frontend_repo" {
  description = "GitHub org/repo for the frontend (S3 sync + CloudFront invalidation)."
  type        = string
  default     = "RidhiD/cuad-ai-demo-fe"
}

variable "github_oidc_branch" {
  description = "Branch ref the OIDC roles trust (e.g. main). Use '*' for any ref."
  type        = string
  default     = "main"
}

###############################################################################
# CloudWatch
###############################################################################

variable "log_retention_days" {
  description = "CloudWatch log retention for the App Runner application log group."
  type        = number
  default     = 14
}

variable "alarm_email" {
  description = <<-EOT
    Optional email for an SNS subscription to receive the App Runner 5xx alarm.
    Leave empty to create the alarm with no notification action.
  EOT
  type        = string
  default     = ""
}
