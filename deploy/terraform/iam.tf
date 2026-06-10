###############################################################################
# IAM — App Runner access role (ECR pull) + instance role (SSM/KMS/SageMaker)
#       and GitHub Actions OIDC roles (backend + frontend)
###############################################################################

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id

  # AWS-managed `aws/ssm` KMS key ARN (alias form is accepted in kms:Decrypt).
  ssm_managed_key_arn = "arn:aws:kms:${var.aws_region}:${local.account_id}:alias/aws/ssm"
}

###############################################################################
# App Runner ACCESS role — lets App Runner pull the image from ECR
###############################################################################

data "aws_iam_policy_document" "apprunner_access_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_access" {
  name               = "${var.project_name}-apprunner-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_access_assume.json
}

resource "aws_iam_role_policy_attachment" "apprunner_access_ecr" {
  role       = aws_iam_role.apprunner_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

###############################################################################
# App Runner INSTANCE role — runtime permissions for the container
#   - ssm:GetParameters on the two parameter ARNs
#   - kms:Decrypt on the aws/ssm managed key
#   - sagemaker:InvokeEndpoint on the two endpoint ARNs
###############################################################################

data "aws_iam_policy_document" "apprunner_instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.project_name}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.apprunner_instance_assume.json
}

data "aws_iam_policy_document" "apprunner_instance" {
  statement {
    sid    = "ReadSsmSecureParams"
    effect = "Allow"
    actions = [
      "ssm:GetParameters",
      "ssm:GetParameter"
    ]
    resources = [
      aws_ssm_parameter.hf_token.arn,
      aws_ssm_parameter.qdrant_api_key.arn
    ]
  }

  statement {
    sid       = "DecryptSsmManagedKey"
    effect    = "Allow"
    actions   = ["kms:Decrypt"]
    resources = [local.ssm_managed_key_arn]
  }

  statement {
    sid    = "InvokeSageMakerEndpoints"
    effect = "Allow"
    actions = [
      "sagemaker:InvokeEndpoint"
    ]
    resources = [
      aws_sagemaker_endpoint.embedder.arn,
      aws_sagemaker_endpoint.reranker.arn
    ]
  }
}

resource "aws_iam_role_policy" "apprunner_instance" {
  name   = "${var.project_name}-apprunner-instance-policy"
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.apprunner_instance.json
}

###############################################################################
# GitHub Actions OIDC provider (optional create)
###############################################################################

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# Resolve the provider ARN whether we created it or it already exists.
data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

###############################################################################
# OIDC role — BACKEND repo: push to ECR + trigger App Runner deploy
###############################################################################

data "aws_iam_policy_document" "github_backend_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_backend_repo}:ref:refs/heads/${var.github_oidc_branch}"]
    }
  }
}

resource "aws_iam_role" "github_backend" {
  name               = "${var.project_name}-gha-backend"
  assume_role_policy = data.aws_iam_policy_document.github_backend_assume.json
}

data "aws_iam_policy_document" "github_backend" {
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "EcrPushPull"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart"
    ]
    resources = [aws_ecr_repository.backend.arn]
  }

  statement {
    sid    = "AppRunnerDeploy"
    effect = "Allow"
    actions = [
      "apprunner:StartDeployment",
      "apprunner:DescribeService",
      "apprunner:ListServices",
      "apprunner:ListOperations"
    ]
    resources = [aws_apprunner_service.backend.arn]
  }
}

resource "aws_iam_role_policy" "github_backend" {
  name   = "${var.project_name}-gha-backend-policy"
  role   = aws_iam_role.github_backend.id
  policy = data.aws_iam_policy_document.github_backend.json
}

###############################################################################
# OIDC role — FRONTEND repo: s3 sync + CloudFront invalidation
###############################################################################

data "aws_iam_policy_document" "github_frontend_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_frontend_repo}:ref:refs/heads/${var.github_oidc_branch}"]
    }
  }
}

resource "aws_iam_role" "github_frontend" {
  name               = "${var.project_name}-gha-frontend"
  assume_role_policy = data.aws_iam_policy_document.github_frontend_assume.json
}

data "aws_iam_policy_document" "github_frontend" {
  statement {
    sid    = "S3Sync"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = [
      aws_s3_bucket.frontend.arn,
      "${aws_s3_bucket.frontend.arn}/*"
    ]
  }

  statement {
    sid    = "CloudFrontInvalidate"
    effect = "Allow"
    actions = [
      "cloudfront:CreateInvalidation",
      "cloudfront:GetInvalidation",
      "cloudfront:GetDistribution"
    ]
    resources = [aws_cloudfront_distribution.frontend.arn]
  }
}

resource "aws_iam_role_policy" "github_frontend" {
  name   = "${var.project_name}-gha-frontend-policy"
  role   = aws_iam_role.github_frontend.id
  policy = data.aws_iam_policy_document.github_frontend.json
}
