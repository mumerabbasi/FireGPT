from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.fire_danger_result import FireDangerResult
    from ..models.subgrid_properties import SubgridProperties


T = TypeVar("T", bound="SubgridData")


@_attrs_define
class SubgridData:
    """Data for a single subgrid including its location and assessment.

    Attributes:
        row (int):
        col (int):
        center_lat (float):
        center_lon (float):
        properties (SubgridProperties): Properties extracted for each subgrid.
        fire_danger (FireDangerResult): Fire danger assessment for a subgrid.
    """

    row: int
    col: int
    center_lat: float
    center_lon: float
    properties: "SubgridProperties"
    fire_danger: "FireDangerResult"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = self.row

        col = self.col

        center_lat = self.center_lat

        center_lon = self.center_lon

        properties = self.properties.to_dict()

        fire_danger = self.fire_danger.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "row": row,
                "col": col,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "properties": properties,
                "fire_danger": fire_danger,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.fire_danger_result import FireDangerResult
        from ..models.subgrid_properties import SubgridProperties

        d = dict(src_dict)
        row = d.pop("row")

        col = d.pop("col")

        center_lat = d.pop("center_lat")

        center_lon = d.pop("center_lon")

        properties = SubgridProperties.from_dict(d.pop("properties"))

        fire_danger = FireDangerResult.from_dict(d.pop("fire_danger"))

        subgrid_data = cls(
            row=row,
            col=col,
            center_lat=center_lat,
            center_lon=center_lon,
            properties=properties,
            fire_danger=fire_danger,
        )

        subgrid_data.additional_properties = d
        return subgrid_data

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
