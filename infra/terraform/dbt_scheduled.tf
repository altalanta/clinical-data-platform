# EventBridge Rule for scheduled dbt runs
resource "aws_cloudwatch_event_rule" "dbt_schedule" {
  name        = "${local.name_prefix}-dbt-schedule"
  description = "Trigger dbt runs on schedule"
  
  # Run daily at 2 AM UTC
  schedule_expression = "cron(0 2 * * ? *)"
  
  tags = {
    Name = "${local.name_prefix}-dbt-schedule"
  }
}

# EventBridge Target - ECS Task
resource "aws_cloudwatch_event_target" "dbt_ecs_target" {
  rule      = aws_cloudwatch_event_rule.dbt_schedule.name
  target_id = "dbt-runner-target"
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.eventbridge_ecs.arn
  
  ecs_target {
    task_count          = 1
    task_definition_arn = aws_ecs_task_definition.dbt_runner.arn
    launch_type         = "FARGATE"
    platform_version    = "LATEST"
    
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs_tasks.id]
      assign_public_ip = var.enable_nat_gateway ? false : true
    }
  }
  
  input = jsonencode({
    containerOverrides = [
      {
        name = "dbt-runner"
        command = ["dbt", "run", "--full-refresh"]
        environment = [
          {
            name  = "DBT_OPERATION"
            value = "scheduled_run"
          }
        ]
      }
    ]
  })
}

# IAM Role for EventBridge to execute ECS tasks
resource "aws_iam_role" "eventbridge_ecs" {
  name = "${local.name_prefix}-eventbridge-ecs"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "${local.name_prefix}-eventbridge-ecs-role"
  }
}

# IAM Policy for EventBridge ECS execution
resource "aws_iam_role_policy" "eventbridge_ecs" {
  name = "${local.name_prefix}-eventbridge-ecs-policy"
  role = aws_iam_role.eventbridge_ecs.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:RunTask"
        ]
        Resource = [
          aws_ecs_task_definition.dbt_runner.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = [
          aws_iam_role.ecs_task_execution.arn,
          aws_iam_role.ecs_task.arn
        ]
      }
    ]
  })
}

# CloudWatch Log Group for scheduled dbt runs
resource "aws_cloudwatch_log_group" "dbt_scheduled" {
  name              = "/ecs/${local.name_prefix}/dbt-scheduled"
  retention_in_days = 14
  
  tags = {
    Name = "${local.name_prefix}-dbt-scheduled-logs"
  }
}

# CloudWatch Metric Filter for dbt failures
resource "aws_cloudwatch_log_metric_filter" "dbt_errors" {
  name           = "${local.name_prefix}-dbt-errors"
  log_group_name = aws_cloudwatch_log_group.dbt_scheduled.name
  pattern        = "ERROR"
  
  metric_transformation {
    name      = "DbtErrorCount"
    namespace = "ClinicalPlatform/dbt"
    value     = "1"
  }
}

# CloudWatch Alarm for dbt failures
resource "aws_cloudwatch_metric_alarm" "dbt_failures" {
  alarm_name          = "${local.name_prefix}-dbt-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DbtErrorCount"
  namespace           = "ClinicalPlatform/dbt"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "This metric monitors dbt run failures"
  alarm_actions       = [] # Add SNS topic ARN for notifications
  
  tags = {
    Name = "${local.name_prefix}-dbt-failures-alarm"
  }
}