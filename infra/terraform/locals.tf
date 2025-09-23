locals {
  # Name prefixes
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Common tags merged with environment-specific tags
  tags = merge(var.common_tags, {
    Environment = var.environment
    Region      = var.aws_region
  })
  
  # Availability zones
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
  
  # CIDR blocks
  vpc_cidr = "10.0.0.0/16"
  public_subnet_cidrs = [
    "10.0.1.0/24",
    "10.0.2.0/24"
  ]
  private_subnet_cidrs = [
    "10.0.11.0/24",
    "10.0.12.0/24"
  ]
  
  # S3 bucket names (must be globally unique)
  s3_bucket_raw      = "${local.name_prefix}-raw-${random_string.bucket_suffix.result}"
  s3_bucket_silver   = "${local.name_prefix}-silver-${random_string.bucket_suffix.result}"
  s3_bucket_gold     = "${local.name_prefix}-gold-${random_string.bucket_suffix.result}"
  s3_bucket_lineage  = "${local.name_prefix}-lineage-${random_string.bucket_suffix.result}"
  s3_bucket_artifacts = "${local.name_prefix}-artifacts-${random_string.bucket_suffix.result}"
}

# Random suffix for S3 bucket uniqueness
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Get available AZs
data "aws_availability_zones" "available" {
  state = "available"
}

# Get current AWS account and region
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}