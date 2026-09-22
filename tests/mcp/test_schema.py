"""Tool input schema generation: read-only / deprecated fields are dropped."""

from pydantic import BaseModel, computed_field
import pytest

from warpweft.mcp.schema import tool_input_schema

pytestmark = pytest.mark.unit


class WithComputed(BaseModel):
    query: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def normalized(self) -> str:  # read-only: derived, not an input
        return self.query.lower()


def test_read_only_computed_field_is_omitted() -> None:
    schema = tool_input_schema(WithComputed)
    assert "query" in schema["properties"]
    assert "normalized" not in schema["properties"]


def test_plain_fields_are_kept() -> None:
    class Plain(BaseModel):
        a: int
        b: str = "x"

    schema = tool_input_schema(Plain)
    assert set(schema["properties"]) == {"a", "b"}
