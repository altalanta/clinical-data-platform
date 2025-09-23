#!/bin/bash
set -e

# Bootstrap script for cloud deployment
# Creates and populates AWS Secrets Manager entries

ENVIRONMENT=${1:-"dev"}
PROJECT_NAME="clinical-platform"
REGION=${AWS_REGION:-"us-east-1"}

echo "🚀 Bootstrapping clinical data platform secrets for environment: $ENVIRONMENT"

# Function to generate random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Function to create or update secret
create_or_update_secret() {
    local secret_name=$1
    local secret_value=$2
    
    echo "📝 Creating/updating secret: $secret_name"
    
    # Check if secret exists
    if aws secretsmanager describe-secret --secret-id "$secret_name" --region "$REGION" >/dev/null 2>&1; then
        # Update existing secret
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region "$REGION" >/dev/null
        echo "✅ Updated existing secret: $secret_name"
    else
        # Create new secret
        aws secretsmanager create-secret \
            --name "$secret_name" \
            --description "Auto-generated secret for clinical data platform" \
            --secret-string "$secret_value" \
            --region "$REGION" >/dev/null
        echo "✅ Created new secret: $secret_name"
    fi
}

# Generate passwords
DB_PASSWORD=$(generate_password)
BI_PASSWORD=$(generate_password)
JWT_SECRET=$(generate_password)

# Get Terraform outputs if available
if [ -f "infra/terraform/terraform.tfstate" ]; then
    echo "📋 Reading Terraform outputs..."
    
    # Extract values from Terraform state
    DB_HOST=$(cd infra/terraform && terraform output -raw rds_endpoint 2>/dev/null || echo "localhost")
    DB_PORT=$(cd infra/terraform && terraform output -raw rds_port 2>/dev/null || echo "5432")
    DB_NAME=$(cd infra/terraform && terraform output -raw rds_database_name 2>/dev/null || echo "clinical_platform")
    
    S3_BUCKET_RAW=$(cd infra/terraform && terraform output -raw s3_bucket_raw 2>/dev/null || echo "")
    S3_BUCKET_SILVER=$(cd infra/terraform && terraform output -raw s3_bucket_silver 2>/dev/null || echo "")
    S3_BUCKET_GOLD=$(cd infra/terraform && terraform output -raw s3_bucket_gold 2>/dev/null || echo "")
    S3_BUCKET_LINEAGE=$(cd infra/terraform && terraform output -raw s3_bucket_lineage 2>/dev/null || echo "")
    S3_BUCKET_ARTIFACTS=$(cd infra/terraform && terraform output -raw s3_bucket_artifacts 2>/dev/null || echo "")
else
    echo "⚠️ No Terraform state found. Using default values."
    DB_HOST="localhost"
    DB_PORT="5432"
    DB_NAME="clinical_platform"
    S3_BUCKET_RAW=""
    S3_BUCKET_SILVER=""
    S3_BUCKET_GOLD=""
    S3_BUCKET_LINEAGE=""
    S3_BUCKET_ARTIFACTS=""
fi

# Create database credentials secret
DB_CREDENTIALS=$(cat <<EOF
{
  "DB_HOST": "$DB_HOST",
  "DB_PORT": "$DB_PORT",
  "DB_NAME": "$DB_NAME",
  "DB_USER": "postgres",
  "DB_PASSWORD": "$DB_PASSWORD",
  "DB_URL": "postgresql://postgres:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/database/credentials" "$DB_CREDENTIALS"

# Create BI readonly user credentials
BI_CREDENTIALS=$(cat <<EOF
{
  "DB_HOST": "$DB_HOST",
  "DB_PORT": "$DB_PORT",
  "DB_NAME": "$DB_NAME",
  "DB_USER": "bi_readonly",
  "DB_PASSWORD": "$BI_PASSWORD",
  "DB_URL": "postgresql://bi_readonly:$BI_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/database/bi-readonly" "$BI_CREDENTIALS"

# Create JWT secret
JWT_SECRET_JSON=$(cat <<EOF
{
  "JWT_SECRET": "$JWT_SECRET"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/auth/jwt-secret" "$JWT_SECRET_JSON"

# Create S3 configuration
S3_CONFIG=$(cat <<EOF
{
  "S3_BUCKET_RAW": "$S3_BUCKET_RAW",
  "S3_BUCKET_SILVER": "$S3_BUCKET_SILVER",
  "S3_BUCKET_GOLD": "$S3_BUCKET_GOLD",
  "S3_BUCKET_LINEAGE": "$S3_BUCKET_LINEAGE",
  "S3_BUCKET_ARTIFACTS": "$S3_BUCKET_ARTIFACTS",
  "AWS_REGION": "$REGION"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/storage/s3-config" "$S3_CONFIG"

# Create application configuration
APP_CONFIG=$(cat <<EOF
{
  "ENVIRONMENT": "$ENVIRONMENT",
  "LOG_LEVEL": "INFO",
  "READ_ONLY_MODE": "0",
  "LOG_SCRUB_VALUES": "1",
  "ENABLE_PHI_REDACTION": "1",
  "PROMETHEUS_ENABLED": "1",
  "OPENLINEAGE_URL": "http://marquez.${PROJECT_NAME}-${ENVIRONMENT}.internal:5000",
  "OPENLINEAGE_NAMESPACE": "${PROJECT_NAME}-${ENVIRONMENT}"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/app/config" "$APP_CONFIG"

# Create MLflow configuration
MLFLOW_CONFIG=$(cat <<EOF
{
  "MLFLOW_TRACKING_URI": "http://mlflow.${PROJECT_NAME}-${ENVIRONMENT}.internal:5000",
  "MLFLOW_S3_ENDPOINT_URL": "",
  "MLFLOW_ARTIFACT_BUCKET": "$S3_BUCKET_ARTIFACTS"
}
EOF
)

create_or_update_secret "${PROJECT_NAME}-${ENVIRONMENT}/mlflow/config" "$MLFLOW_CONFIG"

echo ""
echo "🎉 Bootstrap completed successfully!"
echo ""
echo "📋 Summary of created secrets:"
echo "   - Database credentials: ${PROJECT_NAME}-${ENVIRONMENT}/database/credentials"
echo "   - BI readonly user: ${PROJECT_NAME}-${ENVIRONMENT}/database/bi-readonly"
echo "   - JWT secret: ${PROJECT_NAME}-${ENVIRONMENT}/auth/jwt-secret"
echo "   - S3 configuration: ${PROJECT_NAME}-${ENVIRONMENT}/storage/s3-config"
echo "   - App configuration: ${PROJECT_NAME}-${ENVIRONMENT}/app/config"
echo "   - MLflow configuration: ${PROJECT_NAME}-${ENVIRONMENT}/mlflow/config"
echo ""
echo "🔐 Store these passwords securely:"
echo "   - Database password: $DB_PASSWORD"
echo "   - BI user password: $BI_PASSWORD"
echo "   - JWT secret: $JWT_SECRET"
echo ""
echo "⚠️  Next steps:"
echo "   1. Update your GitHub repository secrets with the values above"
echo "   2. Run Terraform apply to create the infrastructure"
echo "   3. Deploy the application using GitHub Actions"
echo ""

# Create setup file for local development
cat > .env.cloud <<EOF
# Cloud environment configuration
ENVIRONMENT=$ENVIRONMENT
AWS_REGION=$REGION
DB_HOST=$DB_HOST
DB_PORT=$DB_PORT
DB_NAME=$DB_NAME
DB_USER=postgres
DB_PASSWORD=$DB_PASSWORD
JWT_SECRET=$JWT_SECRET
S3_BUCKET_RAW=$S3_BUCKET_RAW
S3_BUCKET_SILVER=$S3_BUCKET_SILVER
S3_BUCKET_GOLD=$S3_BUCKET_GOLD
S3_BUCKET_LINEAGE=$S3_BUCKET_LINEAGE
S3_BUCKET_ARTIFACTS=$S3_BUCKET_ARTIFACTS
MLFLOW_TRACKING_URI=http://mlflow.${PROJECT_NAME}-${ENVIRONMENT}.internal:5000
READ_ONLY_MODE=0
LOG_SCRUB_VALUES=1
ENABLE_PHI_REDACTION=1
PROMETHEUS_ENABLED=1
EOF

echo "📁 Created .env.cloud file for local development"
echo ""
echo "✨ Bootstrap complete! Your clinical data platform is ready for cloud deployment."