from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SubgridProperties")


@_attrs_define
class SubgridProperties:
    """Properties extracted for each subgrid.

    Attributes:
        jrc_forest_cover_presence_10m (Union[None, Unset, int]):
        hansen_treecover2000_perc (Union[None, Unset, float]):
        corine_land_cover_code (Union[None, Unset, int]):
        corine_land_cover_description (Union[None, Unset, str]):
        hansen_lossyear (Union[None, Unset, int]):
        u_wind_ms (Union[None, Unset, float]):
        v_wind_ms (Union[None, Unset, float]):
        temperature_k (Union[None, Unset, float]):
        altitude_m (Union[None, Unset, float]):
    """

    jrc_forest_cover_presence_10m: Union[None, Unset, int] = UNSET
    hansen_treecover2000_perc: Union[None, Unset, float] = UNSET
    corine_land_cover_code: Union[None, Unset, int] = UNSET
    corine_land_cover_description: Union[None, Unset, str] = UNSET
    hansen_lossyear: Union[None, Unset, int] = UNSET
    u_wind_ms: Union[None, Unset, float] = UNSET
    v_wind_ms: Union[None, Unset, float] = UNSET
    temperature_k: Union[None, Unset, float] = UNSET
    altitude_m: Union[None, Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        jrc_forest_cover_presence_10m: Union[None, Unset, int]
        if isinstance(self.jrc_forest_cover_presence_10m, Unset):
            jrc_forest_cover_presence_10m = UNSET
        else:
            jrc_forest_cover_presence_10m = self.jrc_forest_cover_presence_10m

        hansen_treecover2000_perc: Union[None, Unset, float]
        if isinstance(self.hansen_treecover2000_perc, Unset):
            hansen_treecover2000_perc = UNSET
        else:
            hansen_treecover2000_perc = self.hansen_treecover2000_perc

        corine_land_cover_code: Union[None, Unset, int]
        if isinstance(self.corine_land_cover_code, Unset):
            corine_land_cover_code = UNSET
        else:
            corine_land_cover_code = self.corine_land_cover_code

        corine_land_cover_description: Union[None, Unset, str]
        if isinstance(self.corine_land_cover_description, Unset):
            corine_land_cover_description = UNSET
        else:
            corine_land_cover_description = self.corine_land_cover_description

        hansen_lossyear: Union[None, Unset, int]
        if isinstance(self.hansen_lossyear, Unset):
            hansen_lossyear = UNSET
        else:
            hansen_lossyear = self.hansen_lossyear

        u_wind_ms: Union[None, Unset, float]
        if isinstance(self.u_wind_ms, Unset):
            u_wind_ms = UNSET
        else:
            u_wind_ms = self.u_wind_ms

        v_wind_ms: Union[None, Unset, float]
        if isinstance(self.v_wind_ms, Unset):
            v_wind_ms = UNSET
        else:
            v_wind_ms = self.v_wind_ms

        temperature_k: Union[None, Unset, float]
        if isinstance(self.temperature_k, Unset):
            temperature_k = UNSET
        else:
            temperature_k = self.temperature_k

        altitude_m: Union[None, Unset, float]
        if isinstance(self.altitude_m, Unset):
            altitude_m = UNSET
        else:
            altitude_m = self.altitude_m

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if jrc_forest_cover_presence_10m is not UNSET:
            field_dict["jrc_forest_cover_presence_10m"] = jrc_forest_cover_presence_10m
        if hansen_treecover2000_perc is not UNSET:
            field_dict["hansen_treecover2000_perc"] = hansen_treecover2000_perc
        if corine_land_cover_code is not UNSET:
            field_dict["corine_land_cover_code"] = corine_land_cover_code
        if corine_land_cover_description is not UNSET:
            field_dict["corine_land_cover_description"] = corine_land_cover_description
        if hansen_lossyear is not UNSET:
            field_dict["hansen_lossyear"] = hansen_lossyear
        if u_wind_ms is not UNSET:
            field_dict["u_wind_ms"] = u_wind_ms
        if v_wind_ms is not UNSET:
            field_dict["v_wind_ms"] = v_wind_ms
        if temperature_k is not UNSET:
            field_dict["temperature_k"] = temperature_k
        if altitude_m is not UNSET:
            field_dict["altitude_m"] = altitude_m

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_jrc_forest_cover_presence_10m(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        jrc_forest_cover_presence_10m = _parse_jrc_forest_cover_presence_10m(
            d.pop("jrc_forest_cover_presence_10m", UNSET)
        )

        def _parse_hansen_treecover2000_perc(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        hansen_treecover2000_perc = _parse_hansen_treecover2000_perc(d.pop("hansen_treecover2000_perc", UNSET))

        def _parse_corine_land_cover_code(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        corine_land_cover_code = _parse_corine_land_cover_code(d.pop("corine_land_cover_code", UNSET))

        def _parse_corine_land_cover_description(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        corine_land_cover_description = _parse_corine_land_cover_description(
            d.pop("corine_land_cover_description", UNSET)
        )

        def _parse_hansen_lossyear(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        hansen_lossyear = _parse_hansen_lossyear(d.pop("hansen_lossyear", UNSET))

        def _parse_u_wind_ms(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        u_wind_ms = _parse_u_wind_ms(d.pop("u_wind_ms", UNSET))

        def _parse_v_wind_ms(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        v_wind_ms = _parse_v_wind_ms(d.pop("v_wind_ms", UNSET))

        def _parse_temperature_k(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        temperature_k = _parse_temperature_k(d.pop("temperature_k", UNSET))

        def _parse_altitude_m(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        altitude_m = _parse_altitude_m(d.pop("altitude_m", UNSET))

        subgrid_properties = cls(
            jrc_forest_cover_presence_10m=jrc_forest_cover_presence_10m,
            hansen_treecover2000_perc=hansen_treecover2000_perc,
            corine_land_cover_code=corine_land_cover_code,
            corine_land_cover_description=corine_land_cover_description,
            hansen_lossyear=hansen_lossyear,
            u_wind_ms=u_wind_ms,
            v_wind_ms=v_wind_ms,
            temperature_k=temperature_k,
            altitude_m=altitude_m,
        )

        subgrid_properties.additional_properties = d
        return subgrid_properties

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
