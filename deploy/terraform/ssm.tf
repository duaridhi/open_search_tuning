###############################################################################
# SSM Parameter Store — SecureString secrets (AWS-managed aws/ssm KMS key)
#
# Two params only: HF_TOKEN and QDRANT_API_KEY. They are encrypted with the
# AWS-managed `aws/ssm` key (we do NOT pass key_id, so SSM uses aws/ssm by
# default). App Runner injects them via RuntimeEnvironmentSecrets.
#
# Secret values are supplied at apply time via the sensitive vars
# `hf_token` / `qdrant_api_key` (e.g. TF_VAR_hf_token=... terraform apply).
# If left empty, a placeholder is written and `ignore_changes` prevents
# Terraform from clobbering a value you later set out-of-band (console/CLI).
###############################################################################

resource "aws_ssm_parameter" "hf_token" {
  name        = var.hf_token_param_name
  description = "HuggingFace token (HF_TOKEN) for cuad-ai-demo backend."
  type        = "SecureString"
  # No key_id => AWS-managed `aws/ssm` key, per the plan.
  value = var.hf_token != "" ? var.hf_token : "REPLACE_ME"

  lifecycle {
    # Don't overwrite a value set out-of-band when the var is left empty.
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "qdrant_api_key" {
  name        = var.qdrant_api_key_param_name
  description = "Qdrant Cloud API key (QDRANT_API_KEY) for cuad-ai-demo backend."
  type        = "SecureString"
  value       = var.qdrant_api_key != "" ? var.qdrant_api_key : "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }
}
