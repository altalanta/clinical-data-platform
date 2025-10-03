# Security Controls Documentation

This document outlines the security controls implemented in the Clinical Data Platform, how they are tested, and how to operate the system securely.

## Security Controls Overview

### 1. Read-Only Mode Enforcement

**Purpose**: Prevents all write operations when the system is in read-only mode for GxP/HIPAA compliance.

**Implementation**: 
- `ReadOnlyModeMiddleware` in `src/clinical_platform/api/middleware.py:22-47`
- Blocks POST, PUT, PATCH, DELETE methods when `READ_ONLY_MODE=1`
- Returns 403 Forbidden with descriptive error message

**Configuration**:
```bash
# Enable read-only mode
export READ_ONLY_MODE=1

# Or in configuration
config.security.read_only_mode = True
```

**Testing**:
- Unit tests: `tests/test_readonly_middleware.py`
- CI validation: `.github/workflows/ci.yml:128-156`
- Manual testing: `make run-readonly` then attempt write operations

### 2. Mandatory Authentication

**Purpose**: Ensures all protected endpoints require valid API key authentication.

**Implementation**:
- `verify_api_key()` function in `src/clinical_platform/api/main.py:55-83`
- Applied to all endpoints except `/health`
- No bypass mechanisms - API key must be configured and valid

**Configuration**:
```bash
# Set API key
export CDP_SECURITY__API_KEY=your-secure-api-key-here

# Or in configuration
config.security.api_key = SecretStr("your-secure-api-key-here")
```

**Usage**:
```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/studies
```

**Testing**:
- Unit tests: `tests/test_auth_required.py`
- Validates 401 responses for missing/invalid keys
- Validates 500 response when API key not configured

### 3. PHI Redaction

**Purpose**: Automatically redacts PHI/PII from all log output to prevent data leaks.

**Implementation**:
- `PHIFilter` class in `src/clinical_platform/logging_utils.py:20-89`
- Comprehensive regex patterns for SSN, email, phone, dates, MRN, etc.
- Applied to all logging handlers automatically

**Patterns Redacted**:
- SSN: `123-45-6789` → `[REDACTED]`
- Email: `patient@example.com` → `[REDACTED]`
- Phone: `555-123-4567` → `[REDACTED]`
- Dates: `2023-03-15`, `12/25/2023` → `[REDACTED]`
- Medical Record Numbers: `MRN-123456` → `[REDACTED]`
- Subject IDs: `SUBJ123456` → `[REDACTED]`
- File paths with PHI keywords → `[REDACTED_PATH]`
- Database connection strings → `[REDACTED]`

**Configuration**:
```bash
# Enable PHI redaction (default: enabled)
export LOG_SCRUB_VALUES=1

# Automatic in read-only mode
export READ_ONLY_MODE=1
```

**Testing**:
- Unit tests: `tests/test_phi_redaction.py`
- Property-based tests for various PHI formats
- CI validation checks logs for unredacted PHI

### 4. Input Validation and SQL Injection Prevention

**Purpose**: Prevents SQL injection and validates all user inputs.

**Implementation**:
- Path validation with regex: `^[A-Za-z0-9_-]+$` for subject IDs
- Pydantic models for request validation
- Parameterized queries only (no string concatenation)
- Length limits and type checking

**Subject ID Validation**:
```python
subject_id: str = Path(..., regex="^[A-Za-z0-9_-]+$")
```

**Blocked Patterns**:
- SQL injection: `'; DROP TABLE--`, `1' OR '1'='1`
- Path traversal: `../../../etc/passwd`
- XSS attempts: `<script>alert('xss')</script>`
- Special characters: `@#%*`

**Testing**:
- Unit tests: `tests/test_input_validation.py`
- Negative tests with injection payloads
- Verification of parameterized queries

### 5. CORS and Rate Limiting

**Purpose**: Controls cross-origin access and prevents abuse.

**CORS Configuration**:
- Default deny-all in production
- Localhost origins allowed in development
- Methods restricted based on read-only mode
- No wildcard headers (`*`)

**Rate Limiting**:
- 60 requests/minute for read operations
- 10 requests/minute for write operations
- Per-client IP tracking
- Proper `Retry-After` headers

**Implementation**:
- `configure_cors()` in `src/clinical_platform/api/middleware.py:213-233`
- `RateLimitMiddleware` in `src/clinical_platform/api/middleware.py:151-210`

**Testing**:
- Unit tests: `tests/test_cors_ratelimit.py`
- Integration tests with authentication

### 6. Secure Configuration Defaults

**Purpose**: Prevents deployment with insecure defaults.

**Enforced Requirements**:
- No weak JWT secrets in production
- API key required for staging/production
- SSL enabled for non-local storage
- Configuration validation on startup

**Implementation**:
- `_validate_security_config()` in `src/clinical_platform/config.py:200-236`
- Startup validation fails fast with clear error messages

**Weak Secrets Blocked**:
- `change-in-production-please`
- `secret`, `password`, `admin`, `test`
- Any secret shorter than 32 characters

### 7. Error Sanitization

**Purpose**: Prevents PHI leakage through error messages.

**Implementation**:
- `safe_http_exception()` in `src/clinical_platform/api/main.py:42-52`
- PHI redaction applied to all error messages
- Generic error messages returned to clients
- Detailed errors logged securely for debugging

## Operating Read-Only Mode

### Enabling Read-Only Mode

**Environment Variable**:
```bash
export READ_ONLY_MODE=1
export LOG_SCRUB_VALUES=1  # Automatically enabled with read-only
```

**Configuration File**:
```yaml
security:
  read_only_mode: true
  enable_pii_redaction: true
```

**Runtime**:
```bash
# Start API in read-only mode
make run-readonly

# Or manually
READ_ONLY_MODE=1 uvicorn clinical_platform.api.main:app
```

### Verification

**Quick Test**:
```bash
# Should return 403
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"AGE": 45, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0}'

# Should return 200
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/health
```

**Comprehensive Test**:
```bash
make security-smoke
```

### Monitoring Read-Only Mode

**Log Indicators**:
- Look for `Read-only mode violation attempted` warnings
- Security config validation messages on startup
- PHI redaction status in startup logs

**Health Check**:
The `/health` endpoint always works regardless of read-only mode to allow monitoring.

## CI/CD Security Gates

### Security Validation Job

The CI pipeline includes a dedicated `security-validation` job that:

1. **Runs security-focused tests**: All tests tagged with security keywords
2. **Tests read-only enforcement**: Starts API in read-only mode and verifies write operations return 403
3. **Tests authentication**: Verifies protected endpoints require valid API keys
4. **Tests PHI redaction**: Generates logs with PHI and verifies redaction
5. **Dependency scanning**: Checks for vulnerable dependencies
6. **Artifact upload**: Saves security test results and logs

### Required Passes

All security validation steps must pass for the build to succeed:
- ✅ All security unit tests pass
- ✅ Read-only mode blocks all write operations (403)
- ✅ Authentication properly enforced (401 without key)
- ✅ PHI redacted in all log output
- ✅ No high-severity dependency vulnerabilities

### Running Locally

```bash
# Run all security tests
pytest -k "readonly or redaction or auth or cors or ratelimit or phi or security"

# Test read-only mode
make run-readonly &
curl -X POST http://localhost:8000/score  # Should get 403

# Test PHI redaction
make test-phi-redaction
```

## Development Guidelines

### Adding New Endpoints

1. **Always require authentication** except for health checks:
   ```python
   @app.post("/new-endpoint")
   def new_endpoint(api_key: str = Depends(verify_api_key)):
   ```

2. **Use safe error handling**:
   ```python
   except Exception as e:
       raise safe_http_exception(500, "Operation failed", e)
   ```

3. **Validate inputs strictly**:
   ```python
   subject_id: str = Path(..., regex="^[A-Za-z0-9_-]+$")
   ```

4. **Use parameterized queries**:
   ```python
   con.execute("SELECT * FROM table WHERE id = ?", [user_input])
   ```

### Adding New Tests

1. **Security tests must be tagged** with relevant keywords
2. **Test both positive and negative cases**
3. **Include property-based tests for PHI patterns**
4. **Test error conditions and edge cases**

### Configuration Changes

1. **Security settings must be validated** in `_validate_security_config()`
2. **Production environments must enforce secure defaults**
3. **Document any new security-relevant settings**

## Risk Register (Post-Implementation)

### Residual Risks

1. **PHI Pattern Coverage**: Current regex patterns may not catch all PHI formats
   - **Mitigation**: Regular review and expansion of patterns
   - **Future**: Integration with formal PII detection libraries

2. **Rate Limiting Bypass**: Simple IP-based rate limiting can be circumvented
   - **Mitigation**: Consider user-based or token-based rate limiting
   - **Future**: Integration with proper API gateway

3. **Dependency Vulnerabilities**: New vulnerabilities may be discovered
   - **Mitigation**: Automated dependency scanning in CI
   - **Future**: Runtime vulnerability monitoring

4. **Configuration Drift**: Security settings may be accidentally disabled
   - **Mitigation**: CI validation and startup checks
   - **Future**: Configuration compliance monitoring

### Follow-up Security Enhancements

1. **Enhanced PHI Detection**: Integrate with dedicated PII/PHI detection libraries
2. **API Gateway**: Move rate limiting and authentication to dedicated gateway
3. **Audit Logging**: Comprehensive audit trail for all security events
4. **Threat Modeling**: Formal threat modeling exercise
5. **Penetration Testing**: Third-party security assessment
6. **Compliance Certification**: HIPAA compliance audit

### Security Contacts

- **Security Issues**: Report via GitHub Security tab
- **Documentation**: Maintain this file with any security changes
- **CI/CD**: Update security validation as new controls are added