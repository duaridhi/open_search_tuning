###############################################################################
# SageMaker Serverless endpoints — embedder + reranker
#
# Packaging strategy (per plan): pull the model from HF Hub at endpoint creation
# using the AWS HuggingFace PyTorch Inference DLC (CPU). The DLC's inference
# toolkit downloads HF_MODEL_ID at container start when no model artifact is
# supplied, so PrimaryContainer.ModelDataUrl is intentionally omitted.
###############################################################################

# Execution role for SageMaker models. Needs to pull the DLC image from the
# AWS DLC ECR account and write logs; HF Hub download is outbound internet.
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${var.project_name}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

# SageMakerFullAccess is broad; for a demo it covers ECR pull (DLC), CloudWatch
# logs, and model/endpoint plumbing. Tighten for production.
resource "aws_iam_role_policy_attachment" "sagemaker_exec" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

###############################################################################
# Embedder
###############################################################################

resource "aws_sagemaker_model" "embedder" {
  name               = "cuad-embedder"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    image = var.hf_dlc_image_uri
    mode  = "SingleModel"

    environment = {
      HF_MODEL_ID = var.embedder_hf_model_id
      HF_TASK     = var.embedder_hf_task
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "embedder" {
  name = "cuad-embedder-serverless-config"

  production_variants {
    variant_name = "default"
    model_name   = aws_sagemaker_model.embedder.name

    serverless_config {
      memory_size_in_mb = var.serverless_memory_mb
      max_concurrency   = var.serverless_max_concurrency
    }
  }
}

resource "aws_sagemaker_endpoint" "embedder" {
  name                 = var.embedder_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.embedder.name
}

###############################################################################
# Reranker
###############################################################################

resource "aws_sagemaker_model" "reranker" {
  name               = "cuad-reranker"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    image = var.hf_dlc_image_uri
    mode  = "SingleModel"

    environment = {
      HF_MODEL_ID = var.reranker_hf_model_id
      HF_TASK     = var.reranker_hf_task
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "reranker" {
  name = "cuad-reranker-serverless-config"

  production_variants {
    variant_name = "default"
    model_name   = aws_sagemaker_model.reranker.name

    serverless_config {
      memory_size_in_mb = var.serverless_memory_mb
      max_concurrency   = var.serverless_max_concurrency
    }
  }
}

resource "aws_sagemaker_endpoint" "reranker" {
  name                 = var.reranker_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.reranker.name
}
