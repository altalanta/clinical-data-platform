# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"
  
  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      
      log_configuration {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs.name
      }
    }
  }
  
  tags = {
    Name = "${local.name_prefix}-ecs-cluster"
  }
}

# CloudWatch Log Group for ECS
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 7
  
  tags = {
    Name = "${local.name_prefix}-ecs-logs"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "${local.name_prefix}-alb"
  }
}

# ALB Target Group for API
resource "aws_lb_target_group" "api" {
  name     = "${local.name_prefix}-api-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }
  
  tags = {
    Name = "${local.name_prefix}-api-target-group"
  }
}

# ALB Listener
resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  
  tags = {
    Name = "${local.name_prefix}-api-listener"
  }
}

# ECS Task Definition for API
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${aws_ecr_repository.api.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
      
      secrets = [
        {
          name      = "DB_HOST"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_HOST::"
        },
        {
          name      = "DB_PORT"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PORT::"
        },
        {
          name      = "DB_NAME"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_NAME::"
        },
        {
          name      = "DB_USER"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_USER::"
        },
        {
          name      = "DB_PASSWORD"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PASSWORD::"
        },
        {
          name      = "JWT_SECRET"
          valueFrom = "${aws_secretsmanager_secret.jwt_secret.arn}:JWT_SECRET::"
        },
        {
          name      = "S3_BUCKET_RAW"
          valueFrom = "${aws_secretsmanager_secret.s3_config.arn}:S3_BUCKET_RAW::"
        },
        {
          name      = "S3_BUCKET_SILVER"
          valueFrom = "${aws_secretsmanager_secret.s3_config.arn}:S3_BUCKET_SILVER::"
        },
        {
          name      = "S3_BUCKET_GOLD"
          valueFrom = "${aws_secretsmanager_secret.s3_config.arn}:S3_BUCKET_GOLD::"
        },
        {
          name      = "ENVIRONMENT"
          valueFrom = "${aws_secretsmanager_secret.app_config.arn}:ENVIRONMENT::"
        }
      ]
      
      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "PORT"
          value = "8000"
        }
      ]
      
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = {
    Name = "${local.name_prefix}-api-task-def"
  }
}

# ECS Service for API
resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  platform_version = "LATEST"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = var.enable_nat_gateway ? false : true
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  
  depends_on = [
    aws_lb_listener.api,
    aws_iam_role_policy.ecs_task_execution
  ]
  
  tags = {
    Name = "${local.name_prefix}-api-service"
  }
}

# ECS Task Definition for dbt Runner
resource "aws_ecs_task_definition" "dbt_runner" {
  family                   = "${local.name_prefix}-dbt-runner"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "dbt-runner"
      image = "${aws_ecr_repository.dbt_runner.repository_url}:latest"
      
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "dbt"
        }
      }
      
      secrets = [
        {
          name      = "DB_HOST"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_HOST::"
        },
        {
          name      = "DB_PORT"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PORT::"
        },
        {
          name      = "DB_NAME"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_NAME::"
        },
        {
          name      = "DB_USER"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_USER::"
        },
        {
          name      = "DB_PASSWORD"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PASSWORD::"
        },
        {
          name      = "S3_BUCKET_SILVER"
          valueFrom = "${aws_secretsmanager_secret.s3_config.arn}:S3_BUCKET_SILVER::"
        },
        {
          name      = "S3_BUCKET_GOLD"
          valueFrom = "${aws_secretsmanager_secret.s3_config.arn}:S3_BUCKET_GOLD::"
        }
      ]
      
      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "DBT_PROFILES_DIR"
          value = "/app"
        }
      ]
    }
  ])
  
  tags = {
    Name = "${local.name_prefix}-dbt-runner-task-def"
  }
}