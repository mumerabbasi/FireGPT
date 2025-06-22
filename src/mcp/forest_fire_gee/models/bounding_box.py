from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BoundingBox")


@_attrs_define
class BoundingBox:
    """Represents the bounding box for the area of interest.

    Attributes:
        top_left_lat (float): Latitude of the top-left corner.
        top_left_lon (float): Longitude of the top-left corner.
        bottom_right_lat (float): Latitude of the bottom-right corner.
        bottom_right_lon (float): Longitude of the bottom-right corner.
    """

    top_left_lat: float
    top_left_lon: float
    bottom_right_lat: float
    bottom_right_lon: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        top_left_lat = self.top_left_lat

        top_left_lon = self.top_left_lon

        bottom_right_lat = self.bottom_right_lat

        bottom_right_lon = self.bottom_right_lon

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "top_left_lat": top_left_lat,
                "top_left_lon": top_left_lon,
                "bottom_right_lat": bottom_right_lat,
                "bottom_right_lon": bottom_right_lon,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        top_left_lat = d.pop("top_left_lat")

        top_left_lon = d.pop("top_left_lon")

        bottom_right_lat = d.pop("bottom_right_lat")

        bottom_right_lon = d.pop("bottom_right_lon")

        bounding_box = cls(
            top_left_lat=top_left_lat,
            top_left_lon=top_left_lon,
            bottom_right_lat=bottom_right_lat,
            bottom_right_lon=bottom_right_lon,
        )

        bounding_box.additional_properties = d
        return bounding_box

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
