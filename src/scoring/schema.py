"""Request and response contracts for the scoring API.

The field limits are input invariants: a request that is impossible for a
real application (negative income, an age of 150, a score above 1) is rejected
with HTTP 422 before it ever reaches the model.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are an error, not ignored

    application_id: str = Field(min_length=1, max_length=64)

    AMT_INCOME_TOTAL: float = Field(gt=0, le=1e9)
    AMT_CREDIT: float = Field(gt=0, le=1e9)
    AMT_ANNUITY: Optional[float] = Field(default=None, gt=0, le=1e8)
    AMT_GOODS_PRICE: Optional[float] = Field(default=None, gt=0, le=1e9)
    DAYS_BIRTH: int = Field(ge=-36500, le=-6570, description="Negative days; 18 to 100 years")
    DAYS_EMPLOYED: int = Field(ge=-25000, le=365243)
    DAYS_ID_PUBLISH: int = Field(ge=-25000, le=0)
    DAYS_REGISTRATION: float = Field(ge=-30000, le=0)
    EXT_SOURCE_1: Optional[float] = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0, le=1)
    CNT_CHILDREN: int = Field(ge=0, le=20)
    CNT_FAM_MEMBERS: Optional[float] = Field(default=None, ge=1, le=25)
    OWN_CAR_AGE: Optional[float] = Field(default=None, ge=0, le=100)
    REGION_RATING_CLIENT: Literal[1, 2, 3]
    NAME_CONTRACT_TYPE: str = Field(max_length=50)
    NAME_INCOME_TYPE: str = Field(max_length=50)
    NAME_EDUCATION_TYPE: str = Field(max_length=50)
    NAME_HOUSING_TYPE: str = Field(max_length=50)
    FLAG_OWN_CAR: Literal["Y", "N"]
    FLAG_OWN_REALTY: Literal["Y", "N"]


class ScoreResponse(BaseModel):
    application_id: str
    probability_of_default: float
    risk_band: str
    decision: str
    model_name: str
    model_version: str
    request_id: str
