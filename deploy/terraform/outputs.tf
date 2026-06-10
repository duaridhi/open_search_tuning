output "apprunner_service_url" {
  description = "Public HTTPS URL of the App Runner backend service."
  value       = "https://${aws_apprunner_service.backend.service_url}"
}

output "apprunner_service_arn" {
  description = "App Runner service ARN (for warm-up CLI commands)."
  value       = aws_apprunner_service.backend.arn
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain (the frontend public URL)."
  value       = aws_cloudfront_distribution.frontend.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (for cache invalidation in CI)."
  value       = aws_cloudfront_distribution.frontend.id
}

output "frontend_bucket_name" {
  description = "S3 bucket name for the SPA (s3 sync target)."
  value       = aws_s3_bucket.frontend.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for the backend image."
  value       = aws_ecr_repository.backend.repository_url
}

output "embedder_endpoint_name" {
  description = "SageMaker serverless embedder endpoint name."
  value       = aws_sagemaker_endpoint.embedder.name
}

output "reranker_endpoint_name" {
  description = "SageMaker serverless reranker endpoint name."
  value       = aws_sagemaker_endpoint.reranker.name
}

output "github_backend_role_arn" {
  description = "GitHub Actions OIDC role ARN for the backend repo (ECR + App Runner)."
  value       = aws_iam_role.github_backend.arn
}

output "github_frontend_role_arn" {
  description = "GitHub Actions OIDC role ARN for the frontend repo (S3 + CloudFront)."
  value       = aws_iam_role.github_frontend.arn
}
