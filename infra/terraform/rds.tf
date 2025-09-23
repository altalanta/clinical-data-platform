# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "${local.name_prefix}-db-subnet-group"
  }
}

# Random password for RDS
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "postgres" {
  identifier     = "${local.name_prefix}-postgres"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_allocated_storage * 2
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "clinical_platform"
  username = "postgres"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  # Backup configuration
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  # Deletion protection for production
  deletion_protection = false
  skip_final_snapshot = true
  
  # Performance insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn
  
  # Parameter group for connection limits and logging
  parameter_group_name = aws_db_parameter_group.postgres.name
  
  tags = {
    Name = "${local.name_prefix}-postgres"
  }
}

# Custom Parameter Group
resource "aws_db_parameter_group" "postgres" {
  family = "postgres15"
  name   = "${local.name_prefix}-postgres-params"
  
  parameter {
    name  = "log_statement"
    value = "all"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"  # Log queries taking > 1 second
  }
  
  parameter {
    name  = "max_connections"
    value = "100"
  }
  
  tags = {
    Name = "${local.name_prefix}-postgres-params"
  }
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${local.name_prefix}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "${local.name_prefix}-rds-monitoring-role"
  }
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# RDS Proxy for connection pooling (optional, adds cost)
# resource "aws_db_proxy" "postgres" {
#   name                   = "${local.name_prefix}-postgres-proxy"
#   engine_family         = "POSTGRESQL"
#   auth {
#     auth_scheme = "SECRETS"
#     secret_arn  = aws_secretsmanager_secret.db_credentials.arn
#   }
#   role_arn               = aws_iam_role.proxy.arn
#   vpc_subnet_ids         = aws_subnet.private[*].id
#   vpc_security_group_ids = [aws_security_group.rds.id]
#   
#   target {
#     db_instance_identifier = aws_db_instance.postgres.id
#   }
#   
#   tags = {
#     Name = "${local.name_prefix}-postgres-proxy"
#   }
# }