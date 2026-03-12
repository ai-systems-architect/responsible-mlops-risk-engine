# outputs.tf — Output Values
# responsible-mlops-risk-engine
#
# Values printed after terraform apply.
# Copy s3_bucket_models → S3_BUCKET in .env
# Copy sagemaker_role_arn → SAGEMAKER_ROLE_ARN in .env

output "s3_bucket_raw" {
  description = "S3 bucket for raw ACS PUMS data"
  value       = aws_s3_bucket.raw_data.bucket
}

output "s3_bucket_processed" {
  description = "S3 bucket for processed features and preprocessing artifacts"
  value       = aws_s3_bucket.processed_data.bucket
}

output "s3_bucket_models" {
  description = "S3 bucket for model artifacts — use this as S3_BUCKET in .env"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN — use this as SAGEMAKER_ROLE_ARN in .env"
  value       = aws_iam_role.sagemaker_execution.arn
}
