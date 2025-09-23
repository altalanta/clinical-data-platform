# Infrastructure as Code (Terraform)

This directory contains Terraform configurations for deploying the clinical data platform to AWS.

## Architecture Overview

- **VPC**: 2 public + 2 private subnets across 2 AZs
- **ECS Fargate**: API service, dbt runner, monitoring stack
- **RDS PostgreSQL**: Database with encryption and monitoring
- **S3**: Data lake buckets (raw/silver/gold/lineage/artifacts)
- **ECR**: Container registries for API, dbt-runner, worker
- **Secrets Manager**: All secrets and configuration
- **CloudWatch**: Centralized logging and monitoring
- **Optional**: Prometheus, Grafana, OpenLineage/Marquez

## Cost Optimization

Default configuration aims for near-zero costs:
- **t3.micro** RDS instance ($13/month)
- **Fargate tasks** with minimal CPU/memory
- **Single-AZ** deployment (no NAT Gateway by default)
- **S3 lifecycle policies** for automatic archiving
- **7-day log retention**

Estimated monthly cost: **$15-25** for development environment.

## Quick Start

### Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. GitHub repository secrets configured (for CI/CD)

### Local Deployment

```bash
# Initialize Terraform
cd infra/terraform
terraform init

# Plan deployment
terraform plan -var="project_name=clinical-platform-dev"

# Apply (creates all infrastructure)
terraform apply -var="project_name=clinical-platform-dev"

# Get important outputs
terraform output api_url
terraform output ecr_repository_api_url
```

### Bootstrap Secrets

After initial deployment, run the bootstrap script:

```bash
# Generate secrets and populate Secrets Manager
../scripts/bootstrap_cloud.sh

# Verify secrets
aws secretsmanager list-secrets --query 'SecretList[?contains(Name, `clinical-platform`)]'
```

## Configuration

### Variables

Key variables in `variables.tf`:

```hcl
# Basic configuration
aws_region     = "us-east-1"
project_name   = "clinical-platform"
environment    = "dev"

# Cost optimization
db_instance_class = "db.t3.micro"  # Upgrade to db.t3.small for production
enable_nat_gateway = false         # Set to true for production

# Optional features
enable_prometheus   = true
enable_grafana     = true
enable_openlineage = true
```

### Environment-Specific Configs

```bash
# Development
terraform apply -var-file="environments/dev.tfvars"

# Staging
terraform apply -var-file="environments/staging.tfvars"

# Production
terraform apply -var-file="environments/prod.tfvars"
```

## Security

### Network Security
- Private subnets for databases and internal services
- Security groups with least-privilege access
- VPC flow logs enabled
- NACLs for additional layer of security

### Data Security
- All S3 buckets encrypted at rest (AES-256)
- RDS encryption enabled
- Secrets Manager for all sensitive data
- IAM roles with minimal permissions

### PHI/HIPAA Considerations
- Data classification tags on all resources
- Encryption in transit and at rest
- Audit logging via CloudTrail
- BI readonly role for analytics access

## Monitoring & Observability

### CloudWatch
- Centralized logging from all ECS services
- Custom metrics and alarms
- Cost and usage monitoring
- Performance insights for RDS

### Prometheus & Grafana (Optional)
- Application metrics scraping
- Custom dashboards for:
  - FastAPI latency/5xx/p95
  - dbt run timings
  - Data freshness SLAs
  - Infrastructure health

### OpenLineage (Optional)
- Data lineage tracking
- Marquez UI for visualization
- Integration with dbt and Python pipelines

## CI/CD Integration

### GitHub Actions OIDC

The infrastructure includes an OIDC provider and role for GitHub Actions:

```yaml
# In your GitHub Actions workflow
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v2
  with:
    role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
    aws-region: us-east-1
```

### Required GitHub Secrets

Set these in your repository settings:

```
AWS_ROLE_ARN=arn:aws:iam::ACCOUNT:role/clinical-platform-dev-github-actions
AWS_REGION=us-east-1
ECR_REGISTRY=ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
```

## State Management

### Local State (Default)
- Terraform state stored locally in `terraform.tfstate`
- Good for development and testing
- **Warning**: Not suitable for team collaboration

### Remote State (Recommended)

Update `main.tf` to use S3 backend:

```hcl
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "clinical-platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}
```

Create the S3 bucket and DynamoDB table:

```bash
# S3 bucket for state
aws s3 mb s3://your-terraform-state-bucket --region us-east-1
aws s3api put-bucket-versioning \
  --bucket your-terraform-state-bucket \
  --versioning-configuration Status=Enabled

# DynamoDB table for locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

## Deployment Strategies

### Blue-Green Deployment
1. Deploy new version to staging environment
2. Run smoke tests and validation
3. Switch traffic using ALB target groups
4. Keep previous version for quick rollback

### Rolling Updates
1. Update ECS service with new task definition
2. ECS automatically replaces tasks one by one
3. Health checks ensure zero-downtime deployment

## Troubleshooting

### Common Issues

**ECS Tasks Not Starting**
```bash
# Check task logs
aws logs get-log-events \
  --log-group-name /ecs/clinical-platform-dev \
  --log-stream-name api/api/TASK_ID

# Check task definition
aws ecs describe-task-definition \
  --task-definition clinical-platform-dev-api
```

**Database Connection Issues**
```bash
# Test connectivity from ECS task
aws ecs run-task \
  --cluster clinical-platform-dev-cluster \
  --task-definition clinical-platform-dev-api:1 \
  --overrides '{"containerOverrides":[{"name":"api","command":["sh","-c","nc -zv $DB_HOST $DB_PORT"]}]}'
```

**Secrets Access Issues**
```bash
# Verify secrets exist
aws secretsmanager get-secret-value \
  --secret-id clinical-platform-dev/database/credentials

# Check IAM permissions
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::ACCOUNT:role/clinical-platform-dev-ecs-task-execution \
  --action-names secretsmanager:GetSecretValue \
  --resource-arns arn:aws:secretsmanager:REGION:ACCOUNT:secret:clinical-platform-dev/database/credentials-*
```

### Disaster Recovery

**Database Backup**
- Automated daily backups with 7-day retention
- Point-in-time recovery available
- Manual snapshots for major changes

**Infrastructure Recovery**
```bash
# Restore from Terraform state
terraform plan -refresh-only
terraform apply -refresh-only

# Recreate resources if needed
terraform destroy
terraform apply
```

## Cleanup

To avoid ongoing charges:

```bash
# Destroy all resources
terraform destroy

# Verify cleanup
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=Project,Values=clinical-data-platform

# Clean up any remaining resources manually
```

## Support

For infrastructure questions:
- Check CloudWatch logs for application issues
- Use AWS Systems Manager Session Manager for ECS task access
- Monitor costs with AWS Cost Explorer
- Set up billing alerts for unexpected charges