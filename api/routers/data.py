"""
Data access and management endpoints with RBAC enforcement.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, validator

from ..auth import TokenData, UserRole
from ..dependencies import (
    require_data_read, 
    require_data_write, 
    require_admin,
    require_bi_readonly,
    check_read_only_mode,
    get_s3_config
)
from ..telemetry import record_data_processing_time, record_validation_error


router = APIRouter(prefix="/data", tags=["data"])


# Data models
class DataRecord(BaseModel):
    """Generic data record model."""
    id: Optional[str] = None
    source: str
    data_type: str
    content: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content structure."""
        if not isinstance(v, dict):
            raise ValueError("Content must be a dictionary")
        return v


class DataQuery(BaseModel):
    """Data query parameters."""
    source: Optional[str] = None
    data_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100
    offset: int = 0
    
    @validator('limit')
    def validate_limit(cls, v):
        """Validate query limit."""
        if v > 1000:
            raise ValueError("Limit cannot exceed 1000 records")
        return v


class DataUpload(BaseModel):
    """Data upload request."""
    source: str
    data_type: str
    records: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('records')
    def validate_records(cls, v):
        """Validate records list."""
        if len(v) > 100:
            raise ValueError("Cannot upload more than 100 records at once")
        return v


class DataValidationResult(BaseModel):
    """Data validation result."""
    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    record_count: int
    validation_timestamp: datetime


class DataSummary(BaseModel):
    """Data summary statistics."""
    total_records: int
    data_sources: List[str]
    data_types: List[str]
    date_range: Dict[str, Optional[datetime]]
    last_updated: datetime


# Sample data store (replace with actual database in production)
data_store = []


@router.get("/", response_model=List[DataRecord])
async def list_data(
    query: DataQuery = Depends(),
    current_user: TokenData = Depends(require_data_read)
):
    """
    List data records with filtering and pagination.
    Requires data read permission.
    """
    import time
    start_time = time.time()
    
    try:
        # Apply filters
        filtered_data = data_store.copy()
        
        if query.source:
            filtered_data = [r for r in filtered_data if r.get("source") == query.source]
        
        if query.data_type:
            filtered_data = [r for r in filtered_data if r.get("data_type") == query.data_type]
        
        if query.start_date:
            filtered_data = [r for r in filtered_data if r.get("created_at", datetime.min) >= query.start_date]
        
        if query.end_date:
            filtered_data = [r for r in filtered_data if r.get("created_at", datetime.max) <= query.end_date]
        
        # Apply pagination
        total = len(filtered_data)
        paginated_data = filtered_data[query.offset:query.offset + query.limit]
        
        # Convert to DataRecord models
        records = []
        for item in paginated_data:
            record = DataRecord(
                id=item.get("id"),
                source=item.get("source", "unknown"),
                data_type=item.get("data_type", "unknown"),
                content=item.get("content", {}),
                created_at=item.get("created_at", datetime.now()),
                updated_at=item.get("updated_at"),
                metadata=item.get("metadata", {})
            )
            records.append(record)
        
        # Record metrics
        duration = time.time() - start_time
        record_data_processing_time("query", query.source or "all", duration)
        
        return records
        
    except Exception as e:
        record_validation_error("data_query", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying data: {str(e)}"
        )


@router.get("/{record_id}", response_model=DataRecord)
async def get_data_record(
    record_id: str = Path(..., description="Record ID"),
    current_user: TokenData = Depends(require_data_read)
):
    """
    Get a specific data record by ID.
    Requires data read permission.
    """
    # Find record
    record = next((r for r in data_store if r.get("id") == record_id), None)
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data record not found"
        )
    
    return DataRecord(
        id=record.get("id"),
        source=record.get("source", "unknown"),
        data_type=record.get("data_type", "unknown"),
        content=record.get("content", {}),
        created_at=record.get("created_at", datetime.now()),
        updated_at=record.get("updated_at"),
        metadata=record.get("metadata", {})
    )


@router.post("/", response_model=Dict[str, Any])
async def create_data_record(
    record: DataRecord,
    current_user: TokenData = Depends(require_data_write),
    _: None = Depends(check_read_only_mode)
):
    """
    Create a new data record.
    Requires data write permission and not in read-only mode.
    """
    import uuid
    
    # Generate ID if not provided
    if not record.id:
        record.id = str(uuid.uuid4())
    
    # Set timestamps
    now = datetime.now()
    record.created_at = now
    record.updated_at = now
    
    # Add to store
    data_store.append(record.dict())
    
    return {
        "id": record.id,
        "message": "Data record created successfully",
        "created_at": record.created_at
    }


@router.post("/upload", response_model=Dict[str, Any])
async def upload_data_batch(
    upload: DataUpload,
    current_user: TokenData = Depends(require_data_write),
    _: None = Depends(check_read_only_mode)
):
    """
    Upload multiple data records in batch.
    Requires data write permission and not in read-only mode.
    """
    import uuid
    import time
    
    start_time = time.time()
    created_ids = []
    
    try:
        for record_data in upload.records:
            # Create record
            record = {
                "id": str(uuid.uuid4()),
                "source": upload.source,
                "data_type": upload.data_type,
                "content": record_data,
                "created_at": datetime.now(),
                "metadata": upload.metadata or {}
            }
            
            data_store.append(record)
            created_ids.append(record["id"])
        
        # Record metrics
        duration = time.time() - start_time
        record_data_processing_time("upload", upload.source, duration)
        
        return {
            "message": f"Successfully uploaded {len(upload.records)} records",
            "record_ids": created_ids,
            "source": upload.source,
            "data_type": upload.data_type
        }
        
    except Exception as e:
        record_validation_error("data_upload", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading data: {str(e)}"
        )


@router.put("/{record_id}", response_model=Dict[str, Any])
async def update_data_record(
    record_id: str,
    record: DataRecord,
    current_user: TokenData = Depends(require_data_write),
    _: None = Depends(check_read_only_mode)
):
    """
    Update an existing data record.
    Requires data write permission and not in read-only mode.
    """
    # Find existing record
    existing_idx = next((i for i, r in enumerate(data_store) if r.get("id") == record_id), None)
    
    if existing_idx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data record not found"
        )
    
    # Update record
    record.id = record_id
    record.updated_at = datetime.now()
    data_store[existing_idx] = record.dict()
    
    return {
        "id": record_id,
        "message": "Data record updated successfully",
        "updated_at": record.updated_at
    }


@router.delete("/{record_id}")
async def delete_data_record(
    record_id: str,
    current_user: TokenData = Depends(require_admin),
    _: None = Depends(check_read_only_mode)
):
    """
    Delete a data record.
    Requires admin permission and not in read-only mode.
    """
    # Find and remove record
    existing_idx = next((i for i, r in enumerate(data_store) if r.get("id") == record_id), None)
    
    if existing_idx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data record not found"
        )
    
    data_store.pop(existing_idx)
    
    return {
        "message": "Data record deleted successfully",
        "deleted_id": record_id
    }


@router.post("/validate", response_model=DataValidationResult)
async def validate_data(
    upload: DataUpload,
    current_user: TokenData = Depends(require_data_read)
):
    """
    Validate data without storing it.
    Requires data read permission.
    """
    import time
    
    start_time = time.time()
    errors = []
    warnings = []
    
    # Basic validation rules
    for i, record in enumerate(upload.records):
        # Check required fields
        if not isinstance(record, dict):
            errors.append({
                "record_index": i,
                "field": "record",
                "error": "Record must be a dictionary"
            })
            continue
        
        # Check for common PHI fields that should be redacted
        phi_fields = ["ssn", "social_security", "date_of_birth", "phone", "email"]
        for field in phi_fields:
            if field in record:
                warnings.append({
                    "record_index": i,
                    "field": field,
                    "warning": f"Potential PHI field '{field}' detected"
                })
        
        # Validate data types based on data_type
        if upload.data_type == "clinical":
            required_fields = ["patient_id", "visit_date"]
            for field in required_fields:
                if field not in record:
                    errors.append({
                        "record_index": i,
                        "field": field,
                        "error": f"Required field '{field}' is missing"
                    })
    
    # Record metrics
    duration = time.time() - start_time
    record_data_processing_time("validation", upload.source, duration)
    
    if errors:
        record_validation_error("data_validation", "validation_failed")
    
    return DataValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        record_count=len(upload.records),
        validation_timestamp=datetime.now()
    )


@router.get("/summary/stats", response_model=DataSummary)
async def get_data_summary(
    current_user: TokenData = Depends(require_bi_readonly)
):
    """
    Get data summary statistics.
    Requires BI readonly permission or higher.
    """
    if not data_store:
        return DataSummary(
            total_records=0,
            data_sources=[],
            data_types=[],
            date_range={"earliest": None, "latest": None},
            last_updated=datetime.now()
        )
    
    # Calculate summary statistics
    sources = list(set(r.get("source", "unknown") for r in data_store))
    data_types = list(set(r.get("data_type", "unknown") for r in data_store))
    
    dates = [r.get("created_at") for r in data_store if r.get("created_at")]
    earliest = min(dates) if dates else None
    latest = max(dates) if dates else None
    
    return DataSummary(
        total_records=len(data_store),
        data_sources=sources,
        data_types=data_types,
        date_range={"earliest": earliest, "latest": latest},
        last_updated=datetime.now()
    )