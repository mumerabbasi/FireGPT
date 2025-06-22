from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.fire_danger_response_subgrids import FireDangerResponseSubgrids
    from ..models.poi_data import POIData


T = TypeVar("T", bound="FireDangerResponse")


@_attrs_define
class FireDangerResponse:
    """Response model for the fire danger assessment API.
    Includes fire danger for subgrids and relevant POIs.

        Attributes:
            total_subgrids_with_data (int):
            total_pois (int):
            total_pois_within_bbox (int):
            total_pois_within_search_buffer (int):
            subgrids (FireDangerResponseSubgrids):
            key_pois (list['POIData']):
            message (str):
    """

    total_subgrids_with_data: int
    total_pois: int
    total_pois_within_bbox: int
    total_pois_within_search_buffer: int
    subgrids: "FireDangerResponseSubgrids"
    key_pois: list["POIData"]
    message: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_subgrids_with_data = self.total_subgrids_with_data

        total_pois = self.total_pois

        total_pois_within_bbox = self.total_pois_within_bbox

        total_pois_within_search_buffer = self.total_pois_within_search_buffer

        subgrids = self.subgrids.to_dict()

        key_pois = []
        for key_pois_item_data in self.key_pois:
            key_pois_item = key_pois_item_data.to_dict()
            key_pois.append(key_pois_item)

        message = self.message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "total_subgrids_with_data": total_subgrids_with_data,
                "total_pois": total_pois,
                "total_pois_within_bbox": total_pois_within_bbox,
                "total_pois_within_search_buffer": total_pois_within_search_buffer,
                "subgrids": subgrids,
                "key_pois": key_pois,
                "message": message,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.fire_danger_response_subgrids import FireDangerResponseSubgrids
        from ..models.poi_data import POIData

        d = dict(src_dict)
        total_subgrids_with_data = d.pop("total_subgrids_with_data")

        total_pois = d.pop("total_pois")

        total_pois_within_bbox = d.pop("total_pois_within_bbox")

        total_pois_within_search_buffer = d.pop("total_pois_within_search_buffer")

        subgrids = FireDangerResponseSubgrids.from_dict(d.pop("subgrids"))

        key_pois = []
        _key_pois = d.pop("key_pois")
        for key_pois_item_data in _key_pois:
            key_pois_item = POIData.from_dict(key_pois_item_data)

            key_pois.append(key_pois_item)

        message = d.pop("message")

        fire_danger_response = cls(
            total_subgrids_with_data=total_subgrids_with_data,
            total_pois=total_pois,
            total_pois_within_bbox=total_pois_within_bbox,
            total_pois_within_search_buffer=total_pois_within_search_buffer,
            subgrids=subgrids,
            key_pois=key_pois,
            message=message,
        )

        fire_danger_response.additional_properties = d
        return fire_danger_response

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
