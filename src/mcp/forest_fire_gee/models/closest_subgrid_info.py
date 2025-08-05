from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ClosestSubgridInfo")


@_attrs_define
class ClosestSubgridInfo:
    """Information about the closest subgrid to a POI.

    Attributes:
        subgrid_id (str):
        distance_m (float):
    """

    subgrid_id: str
    distance_m: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subgrid_id = self.subgrid_id

        distance_m = self.distance_m

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subgrid_id": subgrid_id,
                "distance_m": distance_m,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        subgrid_id = d.pop("subgrid_id")

        distance_m = d.pop("distance_m")

        closest_subgrid_info = cls(
            subgrid_id=subgrid_id,
            distance_m=distance_m,
        )

        closest_subgrid_info.additional_properties = d
        return closest_subgrid_info

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
