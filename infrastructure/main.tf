# main.tf — AWS Infrastructure Resources
# responsible-mlops-risk-engine
#
# Provisions:
#   - S3 buckets      — raw data, processed data, model artifacts
#   - IAM role        — SageMaker execution role with least-privilege policies
#   - CloudWatch      — endpoint availability, error rate, latency alarms
#
# SageMaker endpoint is NOT provisioned here.
# deploy.py creates and destroys it on demand to avoid ~$5/day standing cost.
#
# Usage:
#   terraform init
#   terraform plan  -var="aws_account_id=YOUR_ACCOUNT_ID"
#   terraform apply -var="aws_account_id=YOUR_ACCOUNT_ID"
#   terraform destroy
#
# After apply — copy outputs to .env:
#   S3_BUCKET          = s3_bucket_models output
#   SAGEMAKER_ROLE_ARN = sagemaker_role_arn output

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# =============================================================================
# S3 BUCKETS
# =============================================================================
# Three buckets maintain separation between data lifecycle stages.
# Versioning enabled on model artifacts — supports rollback if a deployed
# model needs to be reverted to a prior version.

resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-raw-data-${var.aws_account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "raw ACS PUMS data from Census API"
  }
}

resource "aws_s3_bucket" "processed_data" {
  bucket = "${var.project_name}-processed-${var.aws_account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "preprocessed features encoders scaler"
  }
}

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-models-${var.aws_account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "trained model artifacts MLflow registry"
  }
}

# Versioning on model artifacts — enables rollback to prior model versions
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Block all public access on all three buckets
resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket                  = aws_s3_bucket.raw_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed_data" {
  bucket                  = aws_s3_bucket.processed_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


# =============================================================================
# IAM — SAGEMAKER EXECUTION ROLE
# =============================================================================
# Least-privilege role scoped to the three project buckets only.
# No wildcard S3 access — each permission is explicitly justified.

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project_name}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

data "aws_iam_policy_document" "sagemaker_s3_access" {
  # Read access to processed data and model artifacts — required for endpoint startup
  statement {
    sid     = "S3ReadModelArtifacts"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket"]

    resources = [
      aws_s3_bucket.model_artifacts.arn,
      "${aws_s3_bucket.model_artifacts.arn}/*",
      aws_s3_bucket.processed_data.arn,
      "${aws_s3_bucket.processed_data.arn}/*",
    ]
  }

  # Default SageMaker S3 bucket — SDK uploads inference scripts here during deployment
  statement {
    sid     = "S3DefaultSageMakerBucket"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]

    resources = [
      "arn:aws:s3:::sagemaker-${var.aws_region}-${var.aws_account_id}",
      "arn:aws:s3:::sagemaker-${var.aws_region}-${var.aws_account_id}/*",
    ]
  }

  # Write access to model artifacts — required for SageMaker training jobs
  statement {
    sid     = "S3WriteModelArtifacts"
    effect  = "Allow"
    actions = ["s3:PutObject", "s3:DeleteObject"]

    resources = ["${aws_s3_bucket.model_artifacts.arn}/*"]
  }

  # CloudWatch Logs — required for SageMaker endpoint and training job logging
  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]

    resources = [
      "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/*",
    ]
  }

  # ECR — required for SageMaker to pull framework containers
  statement {
    sid    = "ECRReadAccess"
    effect = "Allow"

    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]

    resources = ["*"]
  }
}

resource "aws_iam_policy" "sagemaker_s3_access" {
  name        = "${var.project_name}-sagemaker-s3-access"
  description = "Least-privilege S3 and CloudWatch access for SageMaker execution role"
  policy      = data.aws_iam_policy_document.sagemaker_s3_access.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_s3_access.arn
}


# =============================================================================
# CLOUDWATCH ALARMS
# =============================================================================
# Three alarms cover model health in production:
#
#   endpoint_availability — fires if invocations drop to zero (endpoint down)
#   invocation_errors     — fires if >5% of requests return errors
#   model_latency         — fires if p99 latency exceeds 2000ms
#
# Fairness drift metrics are pushed to CloudWatch by drift_monitor.py
# via Evidently AI — no alarm defined here as thresholds are dynamic.

resource "aws_cloudwatch_metric_alarm" "endpoint_availability" {
  alarm_name          = "${var.project_name}-endpoint-availability"
  alarm_description   = "Fires if endpoint invocations drop to zero — endpoint may be down"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Invocations"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  treat_missing_data  = "breaching"

  dimensions = {
    EndpointName = "responsible-risk-engine-prod-v1"
    VariantName  = "AllTraffic"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "invocation_errors" {
  alarm_name          = "${var.project_name}-invocation-errors"
  alarm_description   = "Fires if more than 5% of invocation requests return errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ModelError"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Average"
  threshold           = 0.05
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "responsible-risk-engine-prod-v1"
    VariantName  = "AllTraffic"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "model_latency" {
  alarm_name          = "${var.project_name}-model-latency"
  alarm_description   = "Fires if p99 inference latency exceeds 2000ms"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 300
  extended_statistic  = "p99"
  threshold           = 2000
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "responsible-risk-engine-prod-v1"
    VariantName  = "AllTraffic"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
