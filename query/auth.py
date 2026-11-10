"""
Authentication + role verification stubs.

Replace `authenticate_user` with your real identity provider (OIDC/JWT).
The rest of the query pipeline depends only on the User object below.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from config import TIER_RANK, ALL_ROLES_PSEUDO_ROLE, ORGANIZATIONAL_ROLES


class AuthenticationError(Exception):
    pass


@dataclass(frozen=True)
class User:
    """Frozen deliberately: immutability removes a class of defect where a
    downstream module widens a principal's roles/clearance after
    authentication but before the access predicate is compiled from it."""
    user_id: str
    roles: List[str]
    sensitivity_clearance: str


def _all_departments() -> List[str]:
    """Executive authority = membership in every departmental role, not a
    privileged bypass in the access predicate (which has no special case)."""
    return ORGANIZATIONAL_ROLES + [ALL_ROLES_PSEUDO_ROLE]


_FAKE_USER_DIRECTORY = {
    "demo-token-finance": User("u_fin_1", ["finance", ALL_ROLES_PSEUDO_ROLE], "confidential"),
    "demo-token-hr": User("u_hr_1", ["hr", ALL_ROLES_PSEUDO_ROLE], "confidential"),
    "demo-token-legal": User("u_legal_1", ["legal", ALL_ROLES_PSEUDO_ROLE], "restricted"),
    "demo-token-engineer": User("u_eng_1", ["engineering", ALL_ROLES_PSEUDO_ROLE], "internal"),
    "demo-token-sales": User("u_sales_1", ["sales", ALL_ROLES_PSEUDO_ROLE], "internal"),
    "demo-token-exec": User("u_exec_1", _all_departments(), "restricted"),
}


def authenticate_user(auth_token: str) -> User:
    user = _FAKE_USER_DIRECTORY.get(auth_token)
    if user is None:
        raise AuthenticationError("Invalid or expired credential.")
    return user


def verify_role(user: User, required_roles: Optional[List[str]] = None) -> bool:
    if not required_roles:
        return True
    return any(r in user.roles for r in required_roles) or ALL_ROLES_PSEUDO_ROLE in user.roles


def allowed_sensitivity_tiers(user: User) -> List[str]:
    max_rank = TIER_RANK.get(user.sensitivity_clearance, 0)
    return [tier for tier, rank in TIER_RANK.items() if rank <= max_rank]
