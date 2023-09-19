from typing import TYPE_CHECKING, Mapping, Sequence

from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    JsonSerializable = (
        int
        | float
        | str
        | None
        | bool
        | Sequence["JsonSerializable"]
        | Mapping[str, "JsonSerializable"]
    )
else:
    JsonSerializable = TypeAliasType(
        "JsonSerializable",
        int
        | float
        | str
        | None
        | bool
        | Sequence["JsonSerializable"]
        | Mapping[str, "JsonSerializable"],
    )
