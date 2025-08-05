from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.bounding_box import BoundingBox


T = TypeVar("T", bound="FireDangerRequest")


@_attrs_define
class FireDangerRequest:
    """Request model for the fire danger assessment API.

    Attributes:
        bbox (BoundingBox): Represents the bounding box for the area of interest.
        subgrid_size_m (Union[Unset, int]): Size of each subgrid cell in meters (e.g., 100 for 100x100m). Minimum 10m.
            Default: 100.
        forecast_hours (Union[Unset, int]): Number of hours into the future for the GFS forecast. Minimum 0 hours.
            Default: 3.
        poi_search_buffer_m (Union[Unset, int]): Buffer distance in meters outside the main bounding box to search for
            Points of Interest. A value of 0 means search only within the bbox. Default: 5000.
    """

    bbox: "BoundingBox"
    subgrid_size_m: Union[Unset, int] = 100
    forecast_hours: Union[Unset, int] = 3
    poi_search_buffer_m: Union[Unset, int] = 5000
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        bbox = self.bbox.to_dict()

        subgrid_size_m = self.subgrid_size_m

        forecast_hours = self.forecast_hours

        poi_search_buffer_m = self.poi_search_buffer_m

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "bbox": bbox,
            }
        )
        if subgrid_size_m is not UNSET:
            field_dict["subgrid_size_m"] = subgrid_size_m
        if forecast_hours is not UNSET:
            field_dict["forecast_hours"] = forecast_hours
        if poi_search_buffer_m is not UNSET:
            field_dict["poi_search_buffer_m"] = poi_search_buffer_m

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.bounding_box import BoundingBox

        d = dict(src_dict)
        bbox = BoundingBox.from_dict(d.pop("bbox"))

        subgrid_size_m = d.pop("subgrid_size_m", UNSET)

        forecast_hours = d.pop("forecast_hours", UNSET)

        poi_search_buffer_m = d.pop("poi_search_buffer_m", UNSET)

        fire_danger_request = cls(
            bbox=bbox,
            subgrid_size_m=subgrid_size_m,
            forecast_hours=forecast_hours,
            poi_search_buffer_m=poi_search_buffer_m,
        )

        fire_danger_request.additional_properties = d
        return fire_danger_request

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
