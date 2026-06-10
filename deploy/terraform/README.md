# cuad-ai-demo — AWS Terraform

Apply-ready Terraform implementing the architecture in
[`readme_docs/AWS_DEPLOYMENT_PLAN.md`](../../readme_docs/AWS_DEPLOYMENT_PLAN.md).

Region: **us-east-1**. Demo-grade, cost-aware (~$1–2/mo). All resources are
serverless / scale-to-zero where possible.

## What gets created

| File | Resources |
|---|---|
| `versions.tf` | Terraform >= 1.5, AWS provider ~> 5.0, default tags |
| `ecr.tf` | ECR repo + lifecycle policy (keep last 10 images) |
| `ssm.tf` | Two SSM SecureString params (`/cuad/hf-token`, `/cuad/qdrant-api-key`) on the AWS-managed `aws/ssm` key |
| `sagemaker.tf` | Embedder + reranker SageMaker **Serverless** endpoints (HF DLC CPU, model pulled from HF Hub at creation) + exec role |
| `iam.tf` | App Runner access role (ECR pull), App Runner instance role (SSM/KMS/SageMaker), GitHub OIDC provider + backend/frontend OIDC roles |
| `apprunner.tf` | App Runner service (port 8080) + auto-scaling config; wires all env vars + SSM secrets |
| `frontend.tf` | Private S3 bucket + CloudFront (OAC, HTTPS) for the single-file SPA |
| `cloudwatch.tf` | App Runner application log group + 5xx alarm (+ optional SNS email) |
| `outputs.tf` | Service URL, CloudFront domain, ECR URL, both endpoint names, OIDC role ARNs |

## Prerequisites

- Terraform >= 1.5, AWS provider ~> 5.0.
- AWS credentials for an account/region (us-east-1) with admin-ish permissions
  for the first apply (creates IAM roles, OIDC provider, SageMaker, etc.).
- Docker, to build and push the backend image to ECR.
- A Qdrant Cloud cluster URL + API key, and a HuggingFace token.
- (Optional) Remote state backend (S3 + DynamoDB). This config uses **local
  state** by default — fine for a demo. Add a `backend "s3"` block to
  `versions.tf` if you want remote state.

## Apply order (important — there is an image bootstrap dependency)

App Runner cannot create a healthy service unless an image exists at
`<ecr_repo_url>:<apprunner_image_tag>`. There are two clean ways to handle this:

### Option A — two-phase apply (recommended)

```bash
# 1. Create just the ECR repo first
terraform init
terraform apply -target=aws_ecr_repository.backend

# 2. Build + push the backend image to that repo
ECR_URL=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${ECR_URL%/*}"
docker build -t "$ECR_URL:latest" ../..        # repo root has the Dockerfile
docker push "$ECR_URL:latest"

# 3. Apply everything else (App Runner now finds the image)
export TF_VAR_hf_token='hf_xxx'
export TF_VAR_qdrant_api_key='qdr_xxx'
terraform apply
```

### Option B — point at a placeholder image first

Set `apprunner_image_tag` to a tag you push manually before the full apply, or
temporarily set `image_identifier` to a public placeholder. Option A is cleaner.

## Setting the two secrets

`ssm.tf` writes the secret values from the sensitive variables `hf_token` and
`qdrant_api_key`. Preferred — pass them as environment variables so they never
touch a file:

```bash
export TF_VAR_hf_token='hf_xxxxxxxx'
export TF_VAR_qdrant_api_key='qdr_xxxxxxxx'
terraform apply
```

If you leave them empty, Terraform creates the parameters with the placeholder
value `REPLACE_ME` and adds `ignore_changes = [value]`, so you can set the real
values out-of-band without Terraform reverting them:

```bash
aws ssm put-parameter --name /cuad/hf-token       --type SecureString --overwrite --value 'hf_xxx'
aws ssm put-parameter --name /cuad/qdrant-api-key --type SecureString --overwrite --value 'qdr_xxx'
```

> Because of `ignore_changes`, after switching from env-var-supplied to
> out-of-band values, Terraform will not show drift on the secret value.

## Configure

```bash
cp terraform.tfvars.example terraform.tfvars
# edit cluster_url, frontend_bucket_name (must be globally unique),
# github_*_repo, etc. Do NOT put secrets in this file.
```

## Deploy / destroy

```bash
terraform init
terraform plan
terraform apply       # (with TF_VAR_hf_token / TF_VAR_qdrant_api_key exported)

terraform destroy     # tears everything down; ECR + S3 have force_delete/force_destroy
```

## After first apply

1. `terraform output apprunner_service_url` → inject into the frontend build as
   `VITE_API_BASE_URL` (GitHub secret for the `cuad-ai-demo-fe` repo).
2. `terraform output cloudfront_domain_name` → this is the public app URL. It is
   already wired into App Runner's `ALLOWED_ORIGINS` automatically.
3. `terraform output github_backend_role_arn` / `github_frontend_role_arn` →
   use these as `role-to-assume` in the respective GitHub Actions workflows
   (`aws-actions/configure-aws-credentials@v4`, no long-lived keys).
4. Build the frontend and `aws s3 sync dist/ s3://<frontend_bucket_name>/`, then
   create a CloudFront invalidation for `/*`.

## Demo-day warm-up

The plan's warm-up / revert procedure (provisioned App Runner instance +
real-time reranker endpoint) is operational and intentionally **not** codified
here — run those `aws` CLI steps from `AWS_DEPLOYMENT_PLAN.md` (§ "Demo Day").

## Notes / caveats

- **App Runner instance size**: the plan says 0.5 vCPU / 1 GB, but App Runner
  does not accept 0.5 vCPU. Defaults here are `0.25 vCPU` / `1 GB`. Bump
  `apprunner_cpu` to `"1 vCPU"` if cold-start CPU is tight.
- **App Runner "scale to zero"**: App Runner auto-pauses idle provisioned
  instances (you pay reduced memory-only rate); `min_size` cannot be 0. To stop
  all idle billing, pause the service.
- **HF DLC image**: `hf_dlc_image_uri` defaults to the confirmed current
  us-east-1 CPU inference DLC tag. If AWS publishes a newer tag, override it.
- **SageMaker exec role** uses `AmazonSageMakerFullAccess` for demo simplicity;
  tighten before production.
- **OIDC provider**: set `create_github_oidc_provider = false` if your account
  already has the `token.actions.githubusercontent.com` provider.
