"""Entity type definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class EntityType(str, Enum):
    PERSON_NAME = "PERSON_NAME"
    ORG_NAME = "ORG_NAME"
    ID_CARD = "ID_CARD"
    PHONE = "PHONE"
    CASE_NUMBER = "CASE_NUMBER"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    MONEY = "MONEY"
    BANK_ACCOUNT = "BANK_ACCOUNT"


ENTITY_TYPE_CN = {
    EntityType.PERSON_NAME: "姓名",
    EntityType.ORG_NAME: "机构名",
    EntityType.ID_CARD: "身份证号",
    EntityType.PHONE: "电话号码",
    EntityType.CASE_NUMBER: "案号",
    EntityType.ADDRESS: "地址",
    EntityType.DATE: "日期",
    EntityType.MONEY: "金额",
    EntityType.BANK_ACCOUNT: "银行账号",
}


@dataclass
class DetectedEntity:
    """A detected PII entity with character offsets."""
    start: int
    end: int
    entity_type: EntityType
    text: str

    def __repr__(self):
        return f"<{self.entity_type.value} [{self.start}:{self.end}] \"{self.text}\">"
