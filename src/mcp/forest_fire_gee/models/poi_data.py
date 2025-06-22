from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.closest_subgrid_info import ClosestSubgridInfo


T = TypeVar("T", bound="POIData")


@_attrs_define
class POIData:
    """Details for a single Point of Interest.

    Attributes:
        osm_id (int):
        type_ (str):
        lat (float):
        lon (float):
        is_within_bbox (bool):
        name (Union[None, Unset, str]):
        subtype (Union[None, Unset, str]):
        closest_subgrid (Union['ClosestSubgridInfo', None, Unset]):
        distance_from_bbox_center_m (Union[None, Unset, float]):
    """

    osm_id: int
    type_: str
    lat: float
    lon: float
    is_within_bbox: bool
    name: Union[None, Unset, str] = UNSET
    subtype: Union[None, Unset, str] = UNSET
    closest_subgrid: Union["ClosestSubgridInfo", None, Unset] = UNSET
    distance_from_bbox_center_m: Union[None, Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.closest_subgrid_info import ClosestSubgridInfo

        osm_id = self.osm_id

        type_ = self.type_

        lat = self.lat

        lon = self.lon

        is_within_bbox = self.is_within_bbox

        name: Union[None, Unset, str]
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        subtype: Union[None, Unset, str]
        if isinstance(self.subtype, Unset):
            subtype = UNSET
        else:
            subtype = self.subtype

        closest_subgrid: Union[None, Unset, dict[str, Any]]
        if isinstance(self.closest_subgrid, Unset):
            closest_subgrid = UNSET
        elif isinstance(self.closest_subgrid, ClosestSubgridInfo):
            closest_subgrid = self.closest_subgrid.to_dict()
        else:
            closest_subgrid = self.closest_subgrid

        distance_from_bbox_center_m: Union[None, Unset, float]
        if isinstance(self.distance_from_bbox_center_m, Unset):
            distance_from_bbox_center_m = UNSET
        else:
            distance_from_bbox_center_m = self.distance_from_bbox_center_m

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "osm_id": osm_id,
                "type": type_,
                "lat": lat,
                "lon": lon,
                "is_within_bbox": is_within_bbox,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if subtype is not UNSET:
            field_dict["subtype"] = subtype
        if closest_subgrid is not UNSET:
            field_dict["closest_subgrid"] = closest_subgrid
        if distance_from_bbox_center_m is not UNSET:
            field_dict["distance_from_bbox_center_m"] = distance_from_bbox_center_m

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.closest_subgrid_info import ClosestSubgridInfo

        d = dict(src_dict)
        osm_id = d.pop("osm_id")

        type_ = d.pop("type")

        lat = d.pop("lat")

        lon = d.pop("lon")

        is_within_bbox = d.pop("is_within_bbox")

        def _parse_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_subtype(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        subtype = _parse_subtype(d.pop("subtype", UNSET))

        def _parse_closest_subgrid(data: object) -> Union["ClosestSubgridInfo", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                closest_subgrid_type_0 = ClosestSubgridInfo.from_dict(data)

                return closest_subgrid_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ClosestSubgridInfo", None, Unset], data)

        closest_subgrid = _parse_closest_subgrid(d.pop("closest_subgrid", UNSET))

        def _parse_distance_from_bbox_center_m(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        distance_from_bbox_center_m = _parse_distance_from_bbox_center_m(d.pop("distance_from_bbox_center_m", UNSET))

        poi_data = cls(
            osm_id=osm_id,
            type_=type_,
            lat=lat,
            lon=lon,
            is_within_bbox=is_within_bbox,
            name=name,
            subtype=subtype,
            closest_subgrid=closest_subgrid,
            distance_from_bbox_center_m=distance_from_bbox_center_m,
        )

        poi_data.additional_properties = d
        return poi_data

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
