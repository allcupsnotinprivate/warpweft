"""warpweft.mcp.tool: the ``@tool`` marker and `Icon` - importable without the mcp SDK."""

import pytest

from warpweft.mcp import Icon, ToolMeta, tool
from warpweft.mcp.tool import tool_meta


def test_icon_accepts_https_urls_and_data_uris() -> None:
    assert Icon("https://example.com/i.png").src == "https://example.com/i.png"
    assert Icon("HTTPS://example.com/i.png").mime_type is None
    assert Icon("data:image/png;base64,AA==", mime_type="image/png", theme="dark").theme == "dark"


def test_icon_sizes_become_a_tuple_so_the_icon_stays_hashable() -> None:
    icon = Icon("https://example.com/i.svg", sizes=["any"])
    assert icon.sizes == ("any",)
    assert hash(icon) == hash(Icon("https://example.com/i.svg", sizes=("any",)))


def test_a_bare_size_string_is_one_size() -> None:
    # str is a Sequence[str] too; "48x48" must not become five one-character sizes.
    assert Icon("https://example.com/i.png", sizes="48x48").sizes == ("48x48",)


@pytest.mark.parametrize(
    "src",
    [
        "http://localhost/i.png",
        "file:///i.png",
        "javascript:alert(1)",
        "i.png",
        "",
        "https:",
        "https:///i.png",
        "data:",
    ],
    ids=["http", "file", "javascript", "relative", "empty", "bare-https", "https-no-host", "bare-data"],
)
def test_icon_rejects_every_other_scheme_or_an_empty_source(src: str) -> None:
    # The spec has clients reject these, so a server should not define one.
    with pytest.raises(ValueError, match="https: URL or a data: URI"):
        Icon(src)


def test_icon_rejects_an_unknown_theme() -> None:
    with pytest.raises(ValueError, match="light.*dark"):
        Icon("https://example.com/i.png", theme="sepia")  # type: ignore[arg-type]


def test_tool_carries_its_icons() -> None:
    icon = Icon("https://example.com/i.png")

    @tool(icons=[icon])
    async def branded(self: object) -> int:
        return 1

    @tool
    async def plain(self: object) -> int:
        return 1

    assert tool_meta(branded) == ToolMeta(icons=(icon,))
    assert tool_meta(plain).icons == ()
