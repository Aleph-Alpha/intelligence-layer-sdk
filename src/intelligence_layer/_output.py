from typing import List, Dict
from pydantic import BaseModel, Field


class AuditTrail(BaseModel):
    pass


class Output(BaseModel):
    audit_trail: AuditTrail = Field(description="audit trail log of a process")
