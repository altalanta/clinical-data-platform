# Prometheus Task Definition
resource "aws_ecs_task_definition" "prometheus" {
  count = var.enable_prometheus ? 1 : 0
  
  family                   = "${local.name_prefix}-prometheus"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "prometheus"
      image = "prom/prometheus:latest"
      
      portMappings = [
        {
          containerPort = 9090
          hostPort      = 9090
          protocol      = "tcp"
        }
      ]
      
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "prometheus"
        }
      }
      
      mountPoints = [
        {
          sourceVolume  = "prometheus-config"
          containerPath = "/etc/prometheus"
          readOnly      = true
        }
      ]
      
      command = [
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
        "--web.console.libraries=/etc/prometheus/console_libraries",
        "--web.console.templates=/etc/prometheus/consoles",
        "--storage.tsdb.retention.time=15d",
        "--web.enable-lifecycle"
      ]
      
      healthCheck = {
        command = ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:9090/-/healthy || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
      }
    }
  ])
  
  volume {
    name = "prometheus-config"
    
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.prometheus_config[0].id
      transit_encryption = "ENABLED"
      
      authorization_config {
        access_point_id = aws_efs_access_point.prometheus_config[0].id
        iam             = "ENABLED"
      }
    }
  }
  
  tags = {
    Name = "${local.name_prefix}-prometheus-task-def"
  }
}

# Grafana Task Definition
resource "aws_ecs_task_definition" "grafana" {
  count = var.enable_grafana ? 1 : 0
  
  family                   = "${local.name_prefix}-grafana"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "grafana"
      image = "grafana/grafana:latest"
      
      portMappings = [
        {
          containerPort = 3000
          hostPort      = 3000
          protocol      = "tcp"
        }
      ]
      
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "grafana"
        }
      }
      
      environment = [
        {
          name  = "GF_SECURITY_ADMIN_PASSWORD"
          value = "admin123"  # Change this in production
        },
        {
          name  = "GF_INSTALL_PLUGINS"
          value = "grafana-clock-panel,grafana-simple-json-datasource"
        }
      ]
      
      mountPoints = [
        {
          sourceVolume  = "grafana-storage"
          containerPath = "/var/lib/grafana"
          readOnly      = false
        }
      ]
      
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
      }
    }
  ])
  
  volume {
    name = "grafana-storage"
    
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.grafana_storage[0].id
      transit_encryption = "ENABLED"
      
      authorization_config {
        access_point_id = aws_efs_access_point.grafana_storage[0].id
        iam             = "ENABLED"
      }
    }
  }
  
  tags = {
    Name = "${local.name_prefix}-grafana-task-def"
  }
}

# ECS Services
resource "aws_ecs_service" "prometheus" {
  count = var.enable_prometheus ? 1 : 0
  
  name            = "${local.name_prefix}-prometheus"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.prometheus[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"
  platform_version = "LATEST"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.monitoring.id]
    assign_public_ip = var.enable_nat_gateway ? false : true
  }
  
  tags = {
    Name = "${local.name_prefix}-prometheus-service"
  }
}

resource "aws_ecs_service" "grafana" {
  count = var.enable_grafana ? 1 : 0
  
  name            = "${local.name_prefix}-grafana"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.grafana[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"
  platform_version = "LATEST"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.monitoring.id]
    assign_public_ip = var.enable_nat_gateway ? false : true
  }
  
  tags = {
    Name = "${local.name_prefix}-grafana-service"
  }
}

# Security Group for Monitoring Services
resource "aws_security_group" "monitoring" {
  name_prefix = "${local.name_prefix}-monitoring-"
  vpc_id      = aws_vpc.main.id
  
  # Prometheus
  ingress {
    from_port = 9090
    to_port   = 9090
    protocol  = "tcp"
    self      = true
  }
  
  # Grafana
  ingress {
    from_port = 3000
    to_port   = 3000
    protocol  = "tcp"
    self      = true
  }
  
  # Allow access from ECS tasks
  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${local.name_prefix}-sg-monitoring"
  }
}

# EFS for persistent storage
resource "aws_efs_file_system" "prometheus_config" {
  count = var.enable_prometheus ? 1 : 0
  
  creation_token = "${local.name_prefix}-prometheus-config"
  encrypted      = true
  
  tags = {
    Name = "${local.name_prefix}-prometheus-config"
  }
}

resource "aws_efs_file_system" "grafana_storage" {
  count = var.enable_grafana ? 1 : 0
  
  creation_token = "${local.name_prefix}-grafana-storage"
  encrypted      = true
  
  tags = {
    Name = "${local.name_prefix}-grafana-storage"
  }
}

# EFS Mount Targets
resource "aws_efs_mount_target" "prometheus_config" {
  count = var.enable_prometheus ? length(aws_subnet.private) : 0
  
  file_system_id  = aws_efs_file_system.prometheus_config[0].id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_mount_target" "grafana_storage" {
  count = var.enable_grafana ? length(aws_subnet.private) : 0
  
  file_system_id  = aws_efs_file_system.grafana_storage[0].id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# EFS Access Points
resource "aws_efs_access_point" "prometheus_config" {
  count = var.enable_prometheus ? 1 : 0
  
  file_system_id = aws_efs_file_system.prometheus_config[0].id
  
  posix_user {
    gid = 65534
    uid = 65534
  }
  
  root_directory {
    path = "/prometheus-config"
    
    creation_info {
      owner_gid   = 65534
      owner_uid   = 65534
      permissions = "755"
    }
  }
  
  tags = {
    Name = "${local.name_prefix}-prometheus-config-ap"
  }
}

resource "aws_efs_access_point" "grafana_storage" {
  count = var.enable_grafana ? 1 : 0
  
  file_system_id = aws_efs_file_system.grafana_storage[0].id
  
  posix_user {
    gid = 472
    uid = 472
  }
  
  root_directory {
    path = "/grafana-storage"
    
    creation_info {
      owner_gid   = 472
      owner_uid   = 472
      permissions = "755"
    }
  }
  
  tags = {
    Name = "${local.name_prefix}-grafana-storage-ap"
  }
}

# Security Group for EFS
resource "aws_security_group" "efs" {
  name_prefix = "${local.name_prefix}-efs-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring.id]
  }
  
  tags = {
    Name = "${local.name_prefix}-sg-efs"
  }
}