# OpenLineage/Marquez Service (optional)
resource "aws_ecs_task_definition" "marquez" {
  count = var.enable_openlineage ? 1 : 0
  
  family                   = "${local.name_prefix}-marquez"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "marquez"
      image = "marquezproject/marquez:latest"
      
      portMappings = [
        {
          containerPort = 5000
          hostPort      = 5000
          protocol      = "tcp"
        }
      ]
      
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "marquez"
        }
      }
      
      secrets = [
        {
          name      = "MARQUEZ_DB_HOST"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_HOST::"
        },
        {
          name      = "MARQUEZ_DB_PORT"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PORT::"
        },
        {
          name      = "MARQUEZ_DB_NAME"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_NAME::"
        },
        {
          name      = "MARQUEZ_DB_USER"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_USER::"
        },
        {
          name      = "MARQUEZ_DB_PASSWORD"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DB_PASSWORD::"
        }
      ]
      
      environment = [
        {
          name  = "MARQUEZ_PORT"
          value = "5000"
        },
        {
          name  = "MARQUEZ_ADMIN_PORT"
          value = "5001"
        }
      ]
      
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:5000/api/v1/namespaces || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = {
    Name = "${local.name_prefix}-marquez-task-def"
  }
}

# ECS Service for Marquez
resource "aws_ecs_service" "marquez" {
  count = var.enable_openlineage ? 1 : 0
  
  name            = "${local.name_prefix}-marquez"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.marquez[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"
  platform_version = "LATEST"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = var.enable_nat_gateway ? false : true
  }
  
  service_registries {
    registry_arn = aws_service_discovery_service.marquez[0].arn
  }
  
  depends_on = [
    aws_iam_role_policy.ecs_task_execution
  ]
  
  tags = {
    Name = "${local.name_prefix}-marquez-service"
  }
}

# Service Discovery for internal communication
resource "aws_service_discovery_private_dns_namespace" "main" {
  count = var.enable_openlineage ? 1 : 0
  
  name = "${local.name_prefix}.internal"
  vpc  = aws_vpc.main.id
  
  tags = {
    Name = "${local.name_prefix}-service-discovery"
  }
}

resource "aws_service_discovery_service" "marquez" {
  count = var.enable_openlineage ? 1 : 0
  
  name = "marquez"
  
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main[0].id
    
    dns_records {
      ttl  = 10
      type = "A"
    }
    
    routing_policy = "MULTIVALUE"
  }
  
  health_check_grace_period_seconds = 30
  
  tags = {
    Name = "${local.name_prefix}-marquez-discovery"
  }
}

# Security group rule for Marquez
resource "aws_security_group_rule" "marquez_access" {
  count = var.enable_openlineage ? 1 : 0
  
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5001
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs_tasks.id
  security_group_id        = aws_security_group.ecs_tasks.id
}