# S3 Bucket for Raw Data
resource "aws_s3_bucket" "raw" {
  bucket = local.s3_bucket_raw
  
  tags = {
    Name        = "${local.name_prefix}-raw"
    DataTier    = "raw"
    Environment = var.environment
  }
}

# S3 Bucket for Silver Data
resource "aws_s3_bucket" "silver" {
  bucket = local.s3_bucket_silver
  
  tags = {
    Name        = "${local.name_prefix}-silver"
    DataTier    = "silver"
    Environment = var.environment
  }
}

# S3 Bucket for Gold Data
resource "aws_s3_bucket" "gold" {
  bucket = local.s3_bucket_gold
  
  tags = {
    Name        = "${local.name_prefix}-gold"
    DataTier    = "gold"
    Environment = var.environment
  }
}

# S3 Bucket for Lineage/Metadata
resource "aws_s3_bucket" "lineage" {
  bucket = local.s3_bucket_lineage
  
  tags = {
    Name        = "${local.name_prefix}-lineage"
    DataTier    = "metadata"
    Environment = var.environment
  }
}

# S3 Bucket for Artifacts (models, reports)
resource "aws_s3_bucket" "artifacts" {
  bucket = local.s3_bucket_artifacts
  
  tags = {
    Name        = "${local.name_prefix}-artifacts"
    DataTier    = "artifacts"
    Environment = var.environment
  }
}

# S3 Bucket Configurations
locals {
  buckets = [
    aws_s3_bucket.raw,
    aws_s3_bucket.silver,
    aws_s3_bucket.gold,
    aws_s3_bucket.lineage,
    aws_s3_bucket.artifacts
  ]
}

# Block public access for all buckets
resource "aws_s3_bucket_public_access_block" "buckets" {
  count = length(local.buckets)
  
  bucket = local.buckets[count.index].id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for all buckets
resource "aws_s3_bucket_versioning" "buckets" {
  count = length(local.buckets)
  
  bucket = local.buckets[count.index].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption for all buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "buckets" {
  count = length(local.buckets)
  
  bucket = local.buckets[count.index].id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Lifecycle policies for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id
  
  rule {
    id     = "raw_data_lifecycle"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "silver_gold" {
  count = 2
  
  bucket = [aws_s3_bucket.silver.id, aws_s3_bucket.gold.id][count.index]
  
  rule {
    id     = "processed_data_lifecycle"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 180
      storage_class = "GLACIER"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# S3 bucket notification for data pipeline triggers (optional)
# resource "aws_s3_bucket_notification" "raw_data" {
#   bucket = aws_s3_bucket.raw.id
#   
#   eventbridge = true
#   
#   depends_on = [aws_s3_bucket_public_access_block.buckets]
# }