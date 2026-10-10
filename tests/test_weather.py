"""Weather tool tests. No network: a fake provider for logic, httpx.MockTransport for the HTTP layer."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from app.tools.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    GEOCODING_URL,
    NoDataError,
    OpenMeteoProvider,
    WeatherProviderError,
    get_weather_forecast,
    parse_period,
)
from tests.fakes import FakeWeatherProvider

TODAY = date(2026, 10, 10)


def forecast(city="Tokyo", when="2026-10-12", provider=None):
    return get_weather_forecast(city, when, provider or FakeWeatherProvider(), today=TODAY)


# --- logic with the fake provider ----------------------------------------------------------------


def test_valid_forecast_is_labelled_mock_and_uses_forecast_endpoint():
    provider = FakeWeatherProvider()
    result = forecast(provider=provider)
    assert result["status"] == "ok"
    assert result["data_kind"] == "forecast"
    assert (result["source"], result["is_live"]) == ("mock-test-data", False)
    assert result["temp_range_c"] == [4.0, 12.0]
    assert result["location"]["name"] == "Tokyo"
    assert provider.calls == [("Tokyo", date(2026, 10, 12), date(2026, 10, 12), "forecast")]
    assert "overcast" in result["conditions"]


def test_past_date_uses_archive_as_observed_data():
    provider = FakeWeatherProvider()
    result = forecast(when="2026-01-15", provider=provider)
    assert result["data_kind"] == "observed_historical"
    assert provider.calls[0][3] == "archive"


def test_recent_past_inside_archive_lag_uses_forecast_endpoint():
    provider = FakeWeatherProvider()
    forecast(when="2026-10-08", provider=provider)
    assert provider.calls[0][3] == "forecast"


def test_date_beyond_forecast_range_is_a_labelled_climate_reference():
    provider = FakeWeatherProvider()
    result = forecast(when="2027-03-01", provider=provider)
    assert result["data_kind"] == "climate_reference"
    assert "not a forecast" in " ".join(result["limitations"])
    assert result["reference_period"].startswith("2026-02-26")
    assert provider.calls[0][3] == "archive"


def test_forecast_boundary_is_15_days_ahead():
    assert forecast(when="2026-10-25")["data_kind"] == "forecast"
    assert forecast(when="2026-10-26")["data_kind"] == "climate_reference"


def test_month_only_future_request_explains_limitation():
    result = forecast(when="November")
    assert result["data_kind"] == "climate_reference"
    assert (result["period_start"], result["period_end"]) == ("2026-11-01", "2026-11-30")
    assert result["reference_period"].startswith("2025-11-01")
    assert "Month-only" in result["limitations"][0] and "specific date" in result["limitations"][0]
    assert result["days_with_data"] == 30


def test_current_month_is_still_not_a_forecast():
    assert forecast(when="2026-10")["data_kind"] == "climate_reference"


def test_completed_month_returns_observed_summary():
    result = forecast(when="2026-09")
    assert result["data_kind"] == "observed_historical"
    assert not result["limitations"]


def test_partial_data_is_reported_not_hidden():
    result = forecast(when="2026-11", provider=FakeWeatherProvider(days_available=10))
    assert result["days_with_data"] == 10
    assert any("10 of 30 days" in text for text in result["limitations"])


@pytest.mark.parametrize("bad", ["", "   ", "banana", "2026-02-30", "2026-13", "13/11/2026"])
def test_invalid_dates_are_rejected_without_calling_provider(bad):
    provider = FakeWeatherProvider()
    result = forecast(when=bad, provider=provider)
    assert result["status"] == "error" and result["error_type"] == "invalid_date"
    assert "YYYY-MM-DD" in result["message"]
    assert provider.calls == []


@pytest.mark.parametrize("bad", ["1800-01-01", "2035-01-01"])
def test_dates_outside_supported_range_are_rejected(bad):
    result = forecast(when=bad)
    assert result["error_type"] == "invalid_date" and "supported range" in result["message"]


@pytest.mark.parametrize("bad", ["", "   ", "12345", "<script>", "a" * 101])
def test_invalid_city_is_rejected(bad):
    assert forecast(city=bad)["error_type"] == "invalid_city"


def test_unknown_city():
    result = forecast(city="Atlantis")
    assert result["status"] == "error" and result["error_type"] == "location_not_found"


def test_country_suffix_is_ignored_for_lookup():
    assert forecast(city="Tokyo, Japan")["status"] == "ok"


def test_provider_failure_never_yields_weather_values():
    result = forecast(provider=FakeWeatherProvider(failure=WeatherProviderError("weather service timed out after 10s")))
    assert result["status"] == "error" and "timed out" in result["message"]
    assert not {"conditions", "temp_range_c", "avg_high_c"} & result.keys()


def test_missing_data_error():
    result = forecast(provider=FakeWeatherProvider(failure=NoDataError("no data")))
    assert result["error_type"] == "no_data"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-12-25", (date(2026, 12, 25), date(2026, 12, 25), "day")),
        ("2026-11", (date(2026, 11, 1), date(2026, 11, 30), "month")),
        ("Nov 2026", (date(2026, 11, 1), date(2026, 11, 30), "month")),
        ("november", (date(2026, 11, 1), date(2026, 11, 30), "month")),
        ("March", (date(2027, 3, 1), date(2027, 3, 31), "month")),  # already past this year -> next year
        ("October", (date(2026, 10, 1), date(2026, 10, 31), "month")),
        ("february 2028", (date(2028, 2, 1), date(2028, 2, 29), "month")),
    ],
)
def test_parse_period(text, expected):
    period = parse_period(text, TODAY)
    assert (period.start, period.end, period.granularity) == expected


# --- HTTP layer with httpx.MockTransport ---------------------------------------------------------

GEO_OK = {"results": [{"name": "Tokyo", "country": "Japan", "admin1": "Tokyo", "latitude": 35.69, "longitude": 139.69}]}
DAILY_OK = {"daily": {
    "time": ["2026-10-12"], "temperature_2m_max": [18.5], "temperature_2m_min": [11.0],
    "precipitation_sum": [0.0], "weather_code": [1]}}


def http_provider(handler) -> OpenMeteoProvider:
    return OpenMeteoProvider(timeout_seconds=3, client=httpx.Client(transport=httpx.MockTransport(handler)))


def route(request: httpx.Request, daily=DAILY_OK, geo=GEO_OK) -> httpx.Response:
    url = str(request.url)
    if url.startswith(GEOCODING_URL):
        return httpx.Response(200, json=geo)
    assert url.startswith((FORECAST_URL, ARCHIVE_URL)), url
    return httpx.Response(200, json=daily)


def test_open_meteo_happy_path_parses_and_sends_expected_params():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return route(request)

    result = get_weather_forecast("Tokyo", "2026-10-12", http_provider(handler), today=TODAY)
    assert result["status"] == "ok" and result["source"] == "open-meteo" and result["is_live"] is True
    assert result["temp_range_c"] == [11.0, 18.5] and result["conditions"] == "mild, mainly clear"
    forecast_request = seen[-1]
    assert str(forecast_request.url).startswith(FORECAST_URL)
    assert forecast_request.url.params["start_date"] == "2026-10-12"
    assert "temperature_2m_max" in forecast_request.url.params["daily"]


def test_open_meteo_archive_url_used_for_past_dates():
    seen = []
    daily = {"daily": {**DAILY_OK["daily"], "time": ["2026-01-15"]}}

    def handler(request):
        seen.append(str(request.url))
        return route(request, daily=daily)

    assert get_weather_forecast("Tokyo", "2026-01-15", http_provider(handler), today=TODAY)["status"] == "ok"
    assert seen[-1].startswith(ARCHIVE_URL)


def test_http_timeout_is_reported():
    def handler(request):
        if str(request.url).startswith(GEOCODING_URL):
            return httpx.Response(200, json=GEO_OK)
        raise httpx.ReadTimeout("slow", request=request)

    result = get_weather_forecast("Tokyo", "2026-10-12", http_provider(handler), today=TODAY)
    assert result["status"] == "error" and "timed out" in result["message"]
    assert result["error_type"] == "service_unavailable" and "temp_range_c" not in result


def test_geocoding_timeout_is_reported():
    def handler(request):
        raise httpx.ConnectTimeout("slow", request=request)

    assert "timed out" in get_weather_forecast("Tokyo", "2026-10-12", http_provider(handler), today=TODAY)["message"]


def test_connection_error_is_reported():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    result = get_weather_forecast("Tokyo", "2026-10-12", http_provider(handler), today=TODAY)
    assert result["status"] == "error" and "could not reach" in result["message"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_http_error_status_is_reported(status):
    result = get_weather_forecast(
        "Tokyo", "2026-10-12", http_provider(lambda r: httpx.Response(status)), today=TODAY
    )
    assert result["status"] == "error" and f"HTTP {status}" in result["message"]


def test_non_json_response_is_invalid():
    result = get_weather_forecast(
        "Tokyo", "2026-10-12", http_provider(lambda r: httpx.Response(200, text="<html>")), today=TODAY
    )
    assert result["error_type"] == "invalid_response"


@pytest.mark.parametrize(
    "daily",
    [
        {"daily": {"time": ["2026-10-12"], "temperature_2m_max": [1.0, 2.0], "temperature_2m_min": [0.0],
                   "precipitation_sum": [0.0]}},  # mismatched lengths
        {"daily": {"time": ["not-a-date"], "temperature_2m_max": [1.0], "temperature_2m_min": [0.0],
                   "precipitation_sum": [0.0]}},
        {"daily": {"time": ["2026-10-12"], "temperature_2m_max": [1.0], "temperature_2m_min": [0.0],
                   "precipitation_sum": [0.0], "weather_code": [1, 2]}},  # weather_code length mismatch
        {"unexpected": True},
    ],
)
def test_invalid_response_structure(daily):
    result = get_weather_forecast(
        "Tokyo", "2026-10-12", http_provider(lambda r: route(r, daily=daily)), today=TODAY
    )
    assert result["status"] == "error" and result["error_type"] == "invalid_response"


def test_all_null_values_mean_no_data():
    daily = {"daily": {"time": ["2026-10-12"], "temperature_2m_max": [None], "temperature_2m_min": [None],
                       "precipitation_sum": [None], "weather_code": [None]}}
    result = get_weather_forecast("Tokyo", "2026-10-12", http_provider(lambda r: route(r, daily=daily)), today=TODAY)
    assert result["error_type"] == "no_data"


def test_days_with_null_values_are_skipped_not_invented():
    daily = {"daily": {"time": ["2026-11-01", "2026-11-02"], "temperature_2m_max": [10.0, None],
                       "temperature_2m_min": [2.0, None], "precipitation_sum": [0.0, None]}}
    result = get_weather_forecast("Tokyo", "2026-11", http_provider(lambda r: route(r, daily=daily)), today=TODAY)
    assert result["days_with_data"] == 1 and result["temp_range_c"] == [2.0, 10.0]


def test_unknown_city_via_empty_geocoding_results():
    result = get_weather_forecast(
        "Nowhereville", "2026-10-12", http_provider(lambda r: route(r, geo={})), today=TODAY
    )
    assert result["error_type"] == "location_not_found"


def test_geocoding_result_missing_fields_is_invalid():
    result = get_weather_forecast(
        "Tokyo", "2026-10-12", http_provider(lambda r: route(r, geo={"results": [{"name": "Tokyo"}]})), today=TODAY
    )
    assert result["error_type"] == "invalid_response"
