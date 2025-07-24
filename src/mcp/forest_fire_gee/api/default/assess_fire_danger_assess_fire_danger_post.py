from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.fire_danger_request import FireDangerRequest
from ...models.fire_danger_response import FireDangerResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: FireDangerRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/assess-fire-danger",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[FireDangerResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = FireDangerResponse.from_dict(response.json())

        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[FireDangerResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: FireDangerRequest,
) -> Response[Union[FireDangerResponse, HTTPValidationError]]:
    """Assess forest fire danger and identify key POIs for a given bounding box

     Receives a bounding box and parameters, fetches relevant geospatial data from Earth Engine,
    calculates a simplified fire danger score for each subgrid, and identifies key Points of Interest
    (POIs)
    within and near the bounding box, providing proximity information.

    **Input:**
    - `bbox`: Defines the area of interest using `top_left_lat`, `top_left_lon`,
      `bottom_right_lat`, and `bottom_right_lon`.
    - `subgrid_size_m`: The desired size (in meters) for each square subgrid cell
      (e.g., 100 for 100x100m). Minimum 10m.
    - `forecast_hours`: The number of hours into the future for the GFS weather forecast
      (e.g., 3 for 3 hours from now). Minimum 0 hours.
    - `poi_search_buffer_m`: Buffer distance in meters outside the main bounding box to search for
      Points of Interest. A value of 0 means search only within the bbox.

    **Output:**
    Returns a JSON object containing:
    - Fire danger scores and contributing factors for each valid subgrid.
    - A list of key POIs, including their location, type, whether they are within the
      main bounding box, and their distance to the closest assessed subgrid.

    Args:
        body (FireDangerRequest): Request model for the fire danger assessment API.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FireDangerResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: FireDangerRequest,
) -> Optional[Union[FireDangerResponse, HTTPValidationError]]:
    """Assess forest fire danger and identify key POIs for a given bounding box

     Receives a bounding box and parameters, fetches relevant geospatial data from Earth Engine,
    calculates a simplified fire danger score for each subgrid, and identifies key Points of Interest
    (POIs)
    within and near the bounding box, providing proximity information.

    **Input:**
    - `bbox`: Defines the area of interest using `top_left_lat`, `top_left_lon`,
      `bottom_right_lat`, and `bottom_right_lon`.
    - `subgrid_size_m`: The desired size (in meters) for each square subgrid cell
      (e.g., 100 for 100x100m). Minimum 10m.
    - `forecast_hours`: The number of hours into the future for the GFS weather forecast
      (e.g., 3 for 3 hours from now). Minimum 0 hours.
    - `poi_search_buffer_m`: Buffer distance in meters outside the main bounding box to search for
      Points of Interest. A value of 0 means search only within the bbox.

    **Output:**
    Returns a JSON object containing:
    - Fire danger scores and contributing factors for each valid subgrid.
    - A list of key POIs, including their location, type, whether they are within the
      main bounding box, and their distance to the closest assessed subgrid.

    Args:
        body (FireDangerRequest): Request model for the fire danger assessment API.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FireDangerResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: FireDangerRequest,
) -> Response[Union[FireDangerResponse, HTTPValidationError]]:
    """Assess forest fire danger and identify key POIs for a given bounding box

     Receives a bounding box and parameters, fetches relevant geospatial data from Earth Engine,
    calculates a simplified fire danger score for each subgrid, and identifies key Points of Interest
    (POIs)
    within and near the bounding box, providing proximity information.

    **Input:**
    - `bbox`: Defines the area of interest using `top_left_lat`, `top_left_lon`,
      `bottom_right_lat`, and `bottom_right_lon`.
    - `subgrid_size_m`: The desired size (in meters) for each square subgrid cell
      (e.g., 100 for 100x100m). Minimum 10m.
    - `forecast_hours`: The number of hours into the future for the GFS weather forecast
      (e.g., 3 for 3 hours from now). Minimum 0 hours.
    - `poi_search_buffer_m`: Buffer distance in meters outside the main bounding box to search for
      Points of Interest. A value of 0 means search only within the bbox.

    **Output:**
    Returns a JSON object containing:
    - Fire danger scores and contributing factors for each valid subgrid.
    - A list of key POIs, including their location, type, whether they are within the
      main bounding box, and their distance to the closest assessed subgrid.

    Args:
        body (FireDangerRequest): Request model for the fire danger assessment API.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FireDangerResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: FireDangerRequest,
) -> Optional[Union[FireDangerResponse, HTTPValidationError]]:
    """Assess forest fire danger and identify key POIs for a given bounding box

     Receives a bounding box and parameters, fetches relevant geospatial data from Earth Engine,
    calculates a simplified fire danger score for each subgrid, and identifies key Points of Interest
    (POIs)
    within and near the bounding box, providing proximity information.

    **Input:**
    - `bbox`: Defines the area of interest using `top_left_lat`, `top_left_lon`,
      `bottom_right_lat`, and `bottom_right_lon`.
    - `subgrid_size_m`: The desired size (in meters) for each square subgrid cell
      (e.g., 100 for 100x100m). Minimum 10m.
    - `forecast_hours`: The number of hours into the future for the GFS weather forecast
      (e.g., 3 for 3 hours from now). Minimum 0 hours.
    - `poi_search_buffer_m`: Buffer distance in meters outside the main bounding box to search for
      Points of Interest. A value of 0 means search only within the bbox.

    **Output:**
    Returns a JSON object containing:
    - Fire danger scores and contributing factors for each valid subgrid.
    - A list of key POIs, including their location, type, whether they are within the
      main bounding box, and their distance to the closest assessed subgrid.

    Args:
        body (FireDangerRequest): Request model for the fire danger assessment API.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FireDangerResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
