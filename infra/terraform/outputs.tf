output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "URL of the API endpoint"
  value       = "http://${aws_lb.main.dns_name}"
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.postgres.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.postgres.db_name
}

output "s3_bucket_raw" {
  description = "S3 bucket for raw data"
  value       = aws_s3_bucket.raw.bucket
}

output "s3_bucket_silver" {
  description = "S3 bucket for silver data"
  value       = aws_s3_bucket.silver.bucket
}

output "s3_bucket_gold" {
  description = "S3 bucket for gold data"
  value       = aws_s3_bucket.gold.bucket
}

output "s3_bucket_lineage" {
  description = "S3 bucket for lineage data"
  value       = aws_s3_bucket.lineage.bucket
}

output "s3_bucket_artifacts" {
  description = "S3 bucket for artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_api_url" {
  description = "ECR repository URL for API"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_repository_dbt_runner_url" {
  description = "ECR repository URL for dbt runner"
  value       = aws_ecr_repository.dbt_runner.repository_url
}

output "ecr_repository_worker_url" {
  description = "ECR repository URL for worker"
  value       = aws_ecr_repository.worker.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "secret_db_credentials_arn" {
  description = "ARN of the database credentials secret"
  value       = aws_secretsmanager_secret.db_credentials.arn
  sensitive   = true
}

output "secret_jwt_secret_arn" {
  description = "ARN of the JWT secret"
  value       = aws_secretsmanager_secret.jwt_secret.arn
  sensitive   = true
}

output "secret_s3_config_arn" {
  description = "ARN of the S3 configuration secret"
  value       = aws_secretsmanager_secret.s3_config.arn
}

output "secret_app_config_arn" {
  description = "ARN of the application configuration secret"
  value       = aws_secretsmanager_secret.app_config.arn
}

output "secret_bi_readonly_arn" {
  description = "ARN of the BI readonly credentials secret"
  value       = aws_secretsmanager_secret.bi_readonly.arn
  sensitive   = true
}

output "github_actions_role_arn" {
  description = "ARN of the GitHub Actions IAM role"
  value       = aws_iam_role.github_actions.arn
}

output "prometheus_url" {
  description = "Prometheus URL (if enabled)"
  value       = var.enable_prometheus ? "http://prometheus.${local.name_prefix}.internal:9090" : null
}

output "grafana_url" {
  description = "Grafana URL (if enabled)"
  value       = var.enable_grafana ? "http://grafana.${local.name_prefix}.internal:3000" : null
}

output "marquez_url" {
  description = "Marquez URL (if enabled)"
  value       = var.enable_openlineage ? "http://marquez.${local.name_prefix}.internal:5000" : null
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for ECS"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}