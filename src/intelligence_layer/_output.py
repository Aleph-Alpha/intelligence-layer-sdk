from typing import List, Dict
from pydantic import BaseModel, Field


class AuditTrail(BaseModel):
    pass


class BaseOutput(BaseModel):
    audit_trail: AuditTrail = Field(description="audit trail log of a process")
