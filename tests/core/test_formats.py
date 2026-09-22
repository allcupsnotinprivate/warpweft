"""String formats: schema annotation and validation on component inputs."""

from typing import Annotated

from pydantic import BaseModel, ValidationError
import pytest

from warpweft.core import formats
from warpweft.core.component import AComponent, EmptySettings, describe, invocable
from warpweft.core.formats import (
    Date,
    DateTime,
    Email,
    Format,
    Hostname,
    Ipv4,
    Ipv6,
    JsonPointer,
    Regex,
    Time,
    Uri,
    Uuid,
)

pytestmark = pytest.mark.unit


def test_format_adds_keyword_and_description_to_schema() -> None:
    class M(BaseModel):
        host: Ipv4

    prop = M.model_json_schema()["properties"]["host"]
    assert prop["type"] == "string"
    assert prop["format"] == "ipv4"
    assert prop["description"] == "IPv4 address."


def test_format_validates_the_value() -> None:
    class M(BaseModel):
        host: Ipv4

    assert M(host="192.168.0.1").host == "192.168.0.1"
    with pytest.raises(ValidationError, match="invalid ipv4"):
        M(host="not-an-ip")


def test_format_without_validator_only_annotates() -> None:
    Opaque = Annotated[str, Format("widget-id", "A widget id.")]

    class M(BaseModel):
        wid: Opaque

    assert M(wid="anything").wid == "anything"  # no validator -> accepts all
    assert M.model_json_schema()["properties"]["wid"]["format"] == "widget-id"


def test_format_without_description_adds_no_description() -> None:
    Bare = Annotated[str, Format("bare")]

    class M(BaseModel):
        v: Bare

    prop = M.model_json_schema()["properties"]["v"]
    assert prop["format"] == "bare"
    assert "description" not in prop


def test_custom_format_needs_no_registry() -> None:
    def _check(value: str) -> None:
        if not value.startswith("case-"):
            raise ValueError("expected a case-... id")

    CaseId = Annotated[str, Format("case-id", "Case id.", _check)]

    class M(BaseModel):
        case: CaseId

    assert M(case="case-1").case == "case-1"
    with pytest.raises(ValidationError, match="invalid case-id"):
        M(case="nope")


def test_formats_flow_into_a_component_input_schema() -> None:
    class Lookup(AComponent[EmptySettings, str, dict]):
        name = "lookup"

        @invocable
        async def go(self, host: Ipv4, ticket: Uuid) -> dict[str, str]:
            return {"host": host}

    schema = describe(Lookup).invocables["go"].input_json_schema()
    assert schema["properties"]["host"]["format"] == "ipv4"
    assert schema["properties"]["ticket"]["format"] == "uuid"


@pytest.mark.parametrize(
    ("fmt", "good", "bad"),
    [
        (Date, "2026-09-16", "16/09/2026"),
        (Time, "13:45:00", "25:00:00"),
        (DateTime, "2026-09-16T13:45:00", "not-a-datetime"),
        (Email, "user@example.com", "user@localhost"),
        (Hostname, "host.example.com", "bad host!"),
        (Ipv4, "10.0.0.1", "999.0.0.1"),
        (Ipv6, "2001:db8::1", "gggg::1"),
        (Uri, "https://example.com/path", "not a uri"),
        (Uuid, "550e8400-e29b-41d4-a716-446655440000", "not-a-uuid"),
        (Regex, "a.*b", "a(b"),
        (JsonPointer, "/a/b", "no-leading-slash"),
    ],
)
def test_standard_formats_accept_and_reject(fmt: type, good: str, bad: str) -> None:
    class M(BaseModel):
        v: fmt  # type: ignore[valid-type]

    assert M(v=good).v == good
    with pytest.raises(ValidationError):
        M(v=bad)


def test_every_shipped_alias_is_exported() -> None:
    # each name in __all__ (except Format itself) is a usable Annotated alias
    for name in formats.__all__:
        if name == "Format":
            continue
        assert getattr(formats, name) is not None
