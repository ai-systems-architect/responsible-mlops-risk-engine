# variables.tf — Input Variables
# responsible-mlops-risk-engine
#
# All variable definitions for the infrastructure.
# Values are passed via -var flags or a terraform.tfvars file.
# Never store actual values in source code — aws_account_id is
# passed at plan/apply time and never committed.

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID — used in IAM policy ARNs and S3 bucket names"
  type        = string
}

variable "project_name" {
  description = "Project name — used as prefix for all resource names"
  type        = string
  default     = "responsible-risk-engine"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}
