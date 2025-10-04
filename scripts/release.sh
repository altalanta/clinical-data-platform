#!/bin/bash
set -euo pipefail

# Clinical Data Platform Release Script
# This script automates the release process for the clinical data platform

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Clinical Data Platform Release Script

Usage: $0 [OPTIONS] VERSION

OPTIONS:
    -h, --help          Show this help message
    -p, --prerelease    Mark as pre-release
    -d, --dry-run       Perform a dry run without creating actual release
    -t, --test          Run tests before release
    -b, --build-only    Only build, don't release
    
VERSION:
    Release version in format: v1.0.0, v1.0.0-rc1, etc.

Examples:
    $0 v1.0.0                    # Create stable release
    $0 --prerelease v1.0.0-rc1   # Create pre-release
    $0 --dry-run v1.0.0          # Test release process
    $0 --test --build-only v1.0.0 # Build and test only

EOF
}

# Parse command line arguments
PRERELEASE=false
DRY_RUN=false
RUN_TESTS=false
BUILD_ONLY=false
VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--prerelease)
            PRERELEASE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        v*.*.*)
            VERSION="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate version
if [[ -z "$VERSION" ]]; then
    log_error "Version is required"
    show_help
    exit 1
fi

if ! [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    log_error "Invalid version format. Expected: v1.0.0 or v1.0.0-rc1"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "$PROJECT_ROOT/clinical-data-platform/pyproject.toml" ]]; then
    log_error "This script must be run from the project root directory"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in git docker poetry curl jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again"
        exit 1
    fi
    
    # Check git status
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    # Check if on main branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        log_warning "Not on main/master branch (currently on: $current_branch)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT/clinical-data-platform"
    
    # Install dependencies
    poetry install --with dev,test
    
    # Run test suite
    poetry run pytest --cov=src/clinical_platform --cov-report=term-missing --cov-fail-under=85
    
    # Run security checks
    poetry run bandit -r src/ --configfile pyproject.toml
    
    # Run dependency vulnerability check
    poetry run safety check --requirements <(poetry export --format requirements.txt --output /dev/stdout --without-hashes)
    
    cd "$PROJECT_ROOT"
    log_success "All tests passed"
}

# Update version in files
update_version() {
    log_info "Updating version to $VERSION..."
    
    local version_number="${VERSION#v}"  # Remove 'v' prefix
    
    cd "$PROJECT_ROOT/clinical-data-platform"
    
    # Update version in pyproject.toml
    poetry version "$version_number"
    
    # Update version in __init__.py if it exists
    if [[ -f "src/clinical_platform/__init__.py" ]]; then
        sed -i.bak "s/__version__ = .*/__version__ = \"$version_number\"/" src/clinical_platform/__init__.py
        rm src/clinical_platform/__init__.py.bak
    fi
    
    cd "$PROJECT_ROOT"
    log_success "Version updated to $VERSION"
}

# Build package
build_package() {
    log_info "Building Python package..."
    
    cd "$PROJECT_ROOT/clinical-data-platform"
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info
    
    # Build package
    poetry build
    
    # Check package
    poetry run twine check dist/*
    
    cd "$PROJECT_ROOT"
    log_success "Package built successfully"
}

# Build Docker images
build_docker_images() {
    log_info "Building Docker images..."
    
    local components=("api" "ingest" "analytics" "streamlit")
    
    for component in "${components[@]}"; do
        log_info "Building $component image..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would build: clinical-data-platform-$component:$VERSION"
        else
            docker build \
                -f "Dockerfile.$component" \
                -t "clinical-data-platform-$component:$VERSION" \
                -t "clinical-data-platform-$component:latest" \
                --build-arg VERSION="$VERSION" \
                --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                --build-arg VCS_REF="$(git rev-parse HEAD)" \
                .
        fi
    done
    
    # Build observability image
    log_info "Building observability image..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: clinical-data-platform-observability:$VERSION"
    else
        docker build \
            -f "observability/Dockerfile" \
            -t "clinical-data-platform-observability:$VERSION" \
            -t "clinical-data-platform-observability:latest" \
            --build-arg VERSION="$VERSION" \
            --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --build-arg VCS_REF="$(git rev-parse HEAD)" \
            observability/
    fi
    
    log_success "Docker images built successfully"
}

# Create Git tag and push
create_git_tag() {
    log_info "Creating Git tag..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create tag: $VERSION"
        return 0
    fi
    
    # Commit version changes
    git add .
    git commit -m "chore: bump version to $VERSION" || true
    
    # Create annotated tag
    if [[ "$PRERELEASE" == "true" ]]; then
        git tag -a "$VERSION" -m "Pre-release $VERSION"
    else
        git tag -a "$VERSION" -m "Release $VERSION"
    fi
    
    # Push changes and tags
    git push origin main
    git push origin "$VERSION"
    
    log_success "Git tag created and pushed"
}

# Trigger GitHub Actions release
trigger_github_release() {
    log_info "Triggering GitHub Actions release workflow..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would trigger GitHub Actions workflow"
        return 0
    fi
    
    # Check if GitHub CLI is available
    if command -v gh &> /dev/null; then
        gh workflow run release.yml \
            --field version="$VERSION" \
            --field prerelease="$PRERELEASE"
        log_success "GitHub Actions workflow triggered"
    else
        log_warning "GitHub CLI not available. Please manually trigger the release workflow or push the tag to trigger it automatically."
    fi
}

# Generate release notes
generate_release_notes() {
    log_info "Generating release notes..."
    
    local release_notes_file="$PROJECT_ROOT/RELEASE_NOTES_$VERSION.md"
    
    # Get previous tag
    local previous_tag
    previous_tag=$(git tag --sort=-version:refname | head -1)
    
    cat > "$release_notes_file" << EOF
# Clinical Data Platform $VERSION

## What's Changed

EOF
    
    # Add commit messages since last tag
    if [[ -n "$previous_tag" ]]; then
        git log "$previous_tag..HEAD" --oneline --pretty="format:* %s (%h)" >> "$release_notes_file"
    else
        echo "* Initial release" >> "$release_notes_file"
    fi
    
    cat >> "$release_notes_file" << EOF

## Docker Images

Multi-architecture Docker images are available:

* \`ghcr.io/altalanta/clinical-data-platform-api:$VERSION\`
* \`ghcr.io/altalanta/clinical-data-platform-ingest:$VERSION\`
* \`ghcr.io/altalanta/clinical-data-platform-analytics:$VERSION\`
* \`ghcr.io/altalanta/clinical-data-platform-streamlit:$VERSION\`
* \`ghcr.io/altalanta/clinical-data-platform-observability:$VERSION\`

## Quick Start

\`\`\`bash
# Install the Python package
pip install clinical-data-platform==${VERSION#v}

# Or run with Docker
docker run ghcr.io/altalanta/clinical-data-platform-api:$VERSION
\`\`\`

## Installation

See the [installation guide](https://altalanta.github.io/clinical-data-platform/installation/) for detailed instructions.
EOF
    
    log_success "Release notes generated: $release_notes_file"
}

# Main execution
main() {
    echo "🚀 Clinical Data Platform Release Process"
    echo "========================================="
    echo "Version: $VERSION"
    echo "Pre-release: $PRERELEASE"
    echo "Dry run: $DRY_RUN"
    echo "Run tests: $RUN_TESTS"
    echo "Build only: $BUILD_ONLY"
    echo ""
    
    # Run prerequisite checks
    check_prerequisites
    
    # Run tests if requested or if it's a stable release
    if [[ "$RUN_TESTS" == "true" ]] || [[ "$PRERELEASE" == "false" && "$DRY_RUN" == "false" ]]; then
        run_tests
    fi
    
    # Update version in files
    if [[ "$DRY_RUN" == "false" ]]; then
        update_version
    fi
    
    # Build package
    build_package
    
    # Build Docker images
    build_docker_images
    
    # If build-only mode, stop here
    if [[ "$BUILD_ONLY" == "true" ]]; then
        log_success "Build completed successfully (build-only mode)"
        exit 0
    fi
    
    # Generate release notes
    generate_release_notes
    
    # Create Git tag and push
    create_git_tag
    
    # Trigger GitHub Actions release
    trigger_github_release
    
    echo ""
    log_success "🎉 Release process completed!"
    echo "============================================"
    echo "Version $VERSION has been successfully released."
    echo ""
    echo "Next steps:"
    echo "1. Monitor the GitHub Actions workflow"
    echo "2. Verify Docker images are published"
    echo "3. Check PyPI package availability"
    echo "4. Update documentation if needed"
    echo ""
    echo "GitHub Release: https://github.com/altalanta/clinical-data-platform/releases/tag/$VERSION"
}

# Execute main function
main "$@"