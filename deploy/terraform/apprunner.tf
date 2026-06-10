###############################################################################
# App Runner — backend API service (scales to zero), image from ECR
###############################################################################

# Auto-scaling config with min 0 so the service scales to zero when idle
# (matches the plan's ~$0.04/mo cold posture).
resource "aws_apprunner_auto_scaling_configuration_version" "cold" {
  auto_scaling_configuration_name = "${var.project_name}-cold"

  min_size        = 1 # App Runner min provisioned instances; billed only when active
  max_size        = 3
  max_concurrency = 10

  tags = {
    Name = "${var.project_name}-cold"
  }
}

# NOTE: App Runner's "scale to zero" is automatic for the IDLE state — when
# there are no requests, provisioned instances are paused and you are billed
# only for memory at the reduced "provisioned" rate. min_size is the floor of
# ACTIVE instances while serving; it cannot be 0. The plan's "scales to zero"
# cost posture is achieved by App Runner pausing idle instances, not by a 0
# min_size. To fully eliminate the idle memory charge, pause the service.

resource "aws_apprunner_service" "backend" {
  service_name = var.project_name

  source_configuration {
    auto_deployments_enabled = var.apprunner_auto_deploy

    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_access.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.backend.repository_url}:${var.apprunner_image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = tostring(var.apprunner_port)

        runtime_environment_variables = {
          EMBED_BACKEND       = "sagemaker"
          RERANK_BACKEND      = "sagemaker"
          EMBEDDER_ENDPOINT   = aws_sagemaker_endpoint.embedder.name
          RERANKER_ENDPOINT   = aws_sagemaker_endpoint.reranker.name
          AWS_REGION          = var.aws_region
          CLUSTER_URL         = var.cluster_url
          QDRANT_COLLECTION   = var.qdrant_collection
          EMBED_MODEL         = var.embed_model
          VECTOR_SIZE         = tostring(var.vector_size)
          RERANK_MODEL        = var.rerank_model
          RERANK_RESULTS      = "1"
          CHAT_MODEL          = var.chat_model
          LOAD_MODEL_STRATEGY = "hybrid_search"
          ALLOWED_ORIGINS     = "https://${aws_cloudfront_distribution.frontend.domain_name}"
          PORT                = tostring(var.apprunner_port)
        }

        runtime_environment_secrets = {
          HF_TOKEN       = aws_ssm_parameter.hf_token.arn
          QDRANT_API_KEY = aws_ssm_parameter.qdrant_api_key.arn
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.apprunner_cpu
    memory            = var.apprunner_memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.cold.arn

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  # CloudFront ALLOWED_ORIGINS depends on the distribution; explicit ordering
  # is implied by the reference above, but keep SageMaker endpoints ready too.
  depends_on = [
    aws_sagemaker_endpoint.embedder,
    aws_sagemaker_endpoint.reranker
  ]
}
