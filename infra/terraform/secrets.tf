# Secrets Manager - Database Credentials
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${local.name_prefix}/database/credentials"
  description = "Database credentials for clinical platform"
  
  tags = {
    Name = "${local.name_prefix}-db-credentials"
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  
  secret_string = jsonencode({
    DB_HOST     = aws_db_instance.postgres.endpoint
    DB_PORT     = aws_db_instance.postgres.port
    DB_NAME     = aws_db_instance.postgres.db_name
    DB_USER     = aws_db_instance.postgres.username
    DB_PASSWORD = random_password.db_password.result
    DB_URL      = "postgresql://${aws_db_instance.postgres.username}:${random_password.db_password.result}@${aws_db_instance.postgres.endpoint}:${aws_db_instance.postgres.port}/${aws_db_instance.postgres.db_name}"
  })
}

# Secrets Manager - JWT Secret
resource "random_password" "jwt_secret" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "jwt_secret" {
  name        = "${local.name_prefix}/auth/jwt-secret"
  description = "JWT signing secret for authentication"
  
  tags = {
    Name = "${local.name_prefix}-jwt-secret"
  }
}

resource "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id = aws_secretsmanager_secret.jwt_secret.id
  
  secret_string = jsonencode({
    JWT_SECRET = random_password.jwt_secret.result
  })
}

# Secrets Manager - S3 Configuration
resource "aws_secretsmanager_secret" "s3_config" {
  name        = "${local.name_prefix}/storage/s3-config"
  description = "S3 bucket configuration"
  
  tags = {
    Name = "${local.name_prefix}-s3-config"
  }
}

resource "aws_secretsmanager_secret_version" "s3_config" {
  secret_id = aws_secretsmanager_secret.s3_config.id
  
  secret_string = jsonencode({
    S3_BUCKET_RAW      = aws_s3_bucket.raw.bucket
    S3_BUCKET_SILVER   = aws_s3_bucket.silver.bucket
    S3_BUCKET_GOLD     = aws_s3_bucket.gold.bucket
    S3_BUCKET_LINEAGE  = aws_s3_bucket.lineage.bucket
    S3_BUCKET_ARTIFACTS = aws_s3_bucket.artifacts.bucket
    AWS_REGION         = var.aws_region
  })
}

# Secrets Manager - MLflow Configuration
resource "aws_secretsmanager_secret" "mlflow_config" {
  name        = "${local.name_prefix}/mlflow/config"
  description = "MLflow tracking server configuration"
  
  tags = {
    Name = "${local.name_prefix}-mlflow-config"
  }
}

resource "aws_secretsmanager_secret_version" "mlflow_config" {
  secret_id = aws_secretsmanager_secret.mlflow_config.id
  
  secret_string = jsonencode({
    MLFLOW_TRACKING_URI     = "http://mlflow.${local.name_prefix}.internal:5000"
    MLFLOW_S3_ENDPOINT_URL  = ""
    MLFLOW_ARTIFACT_BUCKET  = aws_s3_bucket.artifacts.bucket
  })
}

# Secrets Manager - Application Configuration
resource "aws_secretsmanager_secret" "app_config" {
  name        = "${local.name_prefix}/app/config"
  description = "Application configuration secrets"
  
  tags = {
    Name = "${local.name_prefix}-app-config"
  }
}

resource "aws_secretsmanager_secret_version" "app_config" {
  secret_id = aws_secretsmanager_secret.app_config.id
  
  secret_string = jsonencode({
    ENVIRONMENT                = var.environment
    LOG_LEVEL                 = "INFO"
    READ_ONLY_MODE           = "0"
    LOG_SCRUB_VALUES         = "1"
    ENABLE_PHI_REDACTION     = "1"
    PROMETHEUS_ENABLED       = var.enable_prometheus ? "1" : "0"
    OPENLINEAGE_URL          = var.enable_openlineage ? "http://marquez.${local.name_prefix}.internal:5000" : ""
    OPENLINEAGE_NAMESPACE    = local.name_prefix
  })
}

# Secrets Manager - BI Readonly User
resource "random_password" "bi_user_password" {
  length  = 16
  special = true
}

resource "aws_secretsmanager_secret" "bi_readonly" {
  name        = "${local.name_prefix}/database/bi-readonly"
  description = "BI readonly user credentials"
  
  tags = {
    Name = "${local.name_prefix}-bi-readonly"
  }
}

resource "aws_secretsmanager_secret_version" "bi_readonly" {
  secret_id = aws_secretsmanager_secret.bi_readonly.id
  
  secret_string = jsonencode({
    DB_HOST     = aws_db_instance.postgres.endpoint
    DB_PORT     = aws_db_instance.postgres.port
    DB_NAME     = aws_db_instance.postgres.db_name
    DB_USER     = "bi_readonly"
    DB_PASSWORD = random_password.bi_user_password.result
    DB_URL      = "postgresql://bi_readonly:${random_password.bi_user_password.result}@${aws_db_instance.postgres.endpoint}:${aws_db_instance.postgres.port}/${aws_db_instance.postgres.db_name}"
  })
}