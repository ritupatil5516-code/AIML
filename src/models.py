from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
class Transaction(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')
    transaction_id: Optional[str] = Field(None, alias="transactionId")
    account_id: Optional[str] = Field(None, alias="accountId")
    person_id: Optional[str] = Field(None, alias="personId")
    transaction_type: Optional[str] = Field(None, alias="transactionType")
    transaction_status: Optional[str] = Field(None, alias="transactionStatus")
    amount: Optional[float] = None
    transaction_date_time: Optional[str] = Field(None, alias="transactionDateTime")
    currency_code: Optional[str] = Field(None, alias="currencyCode")
    merchant_name: Optional[str] = Field(None, alias="merchantName")
    ending_balance: Optional[float] = Field(None, alias="endingBalance")
    @property
    def id(self) -> str:
        return self.transaction_id or ""
