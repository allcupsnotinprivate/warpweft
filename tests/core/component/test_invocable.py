"""The invocable decorator and IO-schema derivation from annotations."""

from typing import Any

from pydantic import BaseModel
import pytest

from warpweft.core.component.invocable import (
    build_input_model,
    build_output_adapter,
    invocable,
    is_invocable,
    policy_override,
)
from warpweft.core.component.policy import Policy
from warpweft.core.context import InvocationContext

pytestmark = pytest.mark.unit


def test_bare_decorator_marks_with_no_policy() -> None:
    @invocable
    async def m(self: Any) -> None: ...

    assert is_invocable(m)
    assert policy_override(m) is None


def test_decorator_with_policy_records_override() -> None:
    p = Policy(chain=("timeout",))

    @invocable(policy=p)
    async def m(self: Any) -> None: ...

    assert is_invocable(m)
    assert policy_override(m) is p


def test_plain_function_is_not_invocable() -> None:
    async def m(self: Any) -> None: ...

    assert not is_invocable(m)
    assert not is_invocable(42)


class Doc(BaseModel):
    id: str


def test_input_model_drops_self_and_context_and_varargs() -> None:
    async def search(
        self: Any,
        query: str,
        ctx: InvocationContext,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[Doc]: ...

    model = build_input_model("comp", "search", search)
    assert set(model.model_fields) == {"query", "limit"}


def test_input_model_marks_required_vs_defaulted() -> None:
    async def m(self: Any, required: str, optional: int = 3) -> None: ...

    model = build_input_model("comp", "m", m)
    assert model.model_fields["required"].is_required()
    assert not model.model_fields["optional"].is_required()
    assert model.model_fields["optional"].default == 3


def test_missing_annotation_falls_back_to_any() -> None:
    async def m(self: Any, x) -> None: ...  # type: ignore[no-untyped-def]

    model = build_input_model("comp", "m", m)
    assert "x" in model.model_fields


def test_output_adapter_reflects_return_annotation() -> None:
    async def m(self: Any) -> list[Doc]: ...

    schema = build_output_adapter(m).json_schema()
    assert schema["type"] == "array"


def test_output_adapter_without_return_is_any() -> None:
    async def m(self: Any): ...  # type: ignore[no-untyped-def]

    # Any -> permissive empty schema, must not raise.
    assert build_output_adapter(m).json_schema() == {}
