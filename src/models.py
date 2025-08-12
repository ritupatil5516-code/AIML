from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


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

class AccountPersonActivity(BaseModel):
    personId: Optional[str] = None
    purchaseAmount: Optional[float] = 0.0
    rewardsBalance: Optional[float] = 0.0
    rewardsEarned: Optional[float] = 0.0
    rewardsRedeemed: Optional[float] = 0.0
    pendingPurchaseAmount: Optional[float] = 0.0
    pendingRedeemRewardBalance: Optional[float] = 0.0

class AccountSummary(BaseModel):
    # minimal, extend as needed
    accountId: Optional[str] = None
    accountNumberLast4: Optional[str] = Field(default=None, alias="accountNumberLast4")
    accountStatus: Optional[str] = None
    accountType: Optional[str] = None
    productType: Optional[str] = None

    openedDate: Optional[str] = None
    closedDate: Optional[str] = None
    lastUpdatedDate: Optional[str] = None

    creditLimit: Optional[float] = 0.0
    availableCredit: Optional[float] = 0.0
    currentBalance: Optional[float] = 0.0
    currentAdjustedBalance: Optional[float] = 0.0
    totalBalance: Optional[float] = 0.0
    remainingBalance: Optional[float] = 0.0
    revolvingCurrentBalance: Optional[float] = 0.0

    minimumDueAmount: Optional[float] = 0.0
    pastDueAmount: Optional[float] = 0.0
    numberOfInstallments: Optional[int] = 0
    highestPriorityStatus: Optional[str] = None
    balanceStatus: Optional[str] = None
    subStatuses: Optional[List[str]] = None
    flags: Optional[List[str]] = None

    paymentDueDate: Optional[str] = None
    paymentDueDateTime: Optional[str] = None
    billingCycleOpenDateTime: Optional[str] = None
    billingCycleCloseDateTime: Optional[str] = None

    currencyCode: Optional[str] = None
    persons: Optional[List[dict]] = None  # keep raw for now