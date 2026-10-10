"""Weather tool: date parsing, provider interface, Open-Meteo client and result summarisation.

Which Open-Meteo data is used for a request (today = D):
  * specific date <= D-5            -> archive API, observed data             (observed_historical)
  * specific date in D-4 .. D+15    -> forecast API                            (forecast)
  * specific date > D+15            -> same dates last year, archive API       (climate_reference)
  * month only, month already over  -> archive API for that month              (observed_historical)
  * month only, current/future      -> same month last year, archive API       (climate_reference)
A climate_reference is last year's observed weather used as a rough guide. It is never a forecast,
and the result says so in `limitations`.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError, model_validator

from app.schemas.responses import DailyWeather, Location, WeatherResult

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_MAX_DAYS_AHEAD = 15  # forecast API serves 16 days including today
ARCHIVE_LAG_DAYS = 5  # archive data trails real time by a few days
EARLIEST_YEAR = 1940  # archive coverage starts here
MAX_FUTURE_YEARS = 2
CLIMATE_WINDOW_DAYS = 3  # +/- days around a far-future date when using last year's data
RAIN_DAY_MM = 1.0

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_CITY_PATTERN = re.compile(r"^[^\W\d_][\w\s.,'’\-()]*$")
_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# --- Errors ------------------------------------------------------------------------------------


class WeatherProviderError(Exception):
    """A provider failed; `error_type` is surfaced to the LLM in the tool result."""

    error_type = "service_unavailable"


class LocationNotFoundError(WeatherProviderError):
    error_type = "location_not_found"


class InvalidProviderResponseError(WeatherProviderError):
    error_type = "invalid_response"


class NoDataError(WeatherProviderError):
    error_type = "no_data"


# --- Provider interface -------------------------------------------------------------------------


class WeatherProvider(Protocol):
    name: str
    is_live: bool

    def geocode(self, city: str) -> Location: ...

    def daily(
        self, location: Location, start: date, end: date, endpoint: Literal["forecast", "archive"]
    ) -> list[DailyWeather]: ...


class _DailyPayload(BaseModel):
    time: list[date]
    temperature_2m_max: list[float | None]
    temperature_2m_min: list[float | None]
    precipitation_sum: list[float | None]
    weather_code: list[int | None] = []

    @model_validator(mode="after")
    def _same_length(self) -> _DailyPayload:
        n = len(self.time)
        lengths = {len(self.temperature_2m_max), len(self.temperature_2m_min), len(self.precipitation_sum)}
        if self.weather_code:
            lengths.add(len(self.weather_code))
        if lengths != {n}:
            raise ValueError("daily arrays have different lengths")
        return self


class OpenMeteoProvider:
    """Live provider. Free and keyless; subject to Open-Meteo's fair-use limits."""

    name = "open-meteo"
    is_live = True

    def __init__(self, timeout_seconds: float = 10.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._timeout = timeout_seconds

    def _get_json(self, url: str, params: dict) -> dict:
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise WeatherProviderError(f"weather service timed out after {self._timeout:g}s") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherProviderError(
                f"weather service returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise WeatherProviderError(f"could not reach weather service ({type(exc).__name__})") from exc
        except ValueError as exc:  # response body was not JSON
            raise InvalidProviderResponseError("weather service returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise InvalidProviderResponseError("weather service returned an unexpected payload")
        return payload

    def geocode(self, city: str) -> Location:
        payload = self._get_json(
            GEOCODING_URL, {"name": city, "count": 1, "language": "en", "format": "json"}
        )
        results = payload.get("results") or []
        if not results:
            raise LocationNotFoundError(f"no location found for {city!r}")
        try:
            top = results[0]
            return Location(
                name=top["name"],
                country=top.get("country", ""),
                region=top.get("admin1", ""),
                latitude=top["latitude"],
                longitude=top["longitude"],
            )
        except (KeyError, ValidationError) as exc:
            raise InvalidProviderResponseError("geocoding response missing required fields") from exc

    def daily(
        self, location: Location, start: date, end: date, endpoint: Literal["forecast", "archive"]
    ) -> list[DailyWeather]:
        url = FORECAST_URL if endpoint == "forecast" else ARCHIVE_URL
        payload = self._get_json(
            url,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "auto",
            },
        )
        try:
            daily = _DailyPayload.model_validate(payload.get("daily") or {})
        except ValidationError as exc:
            raise InvalidProviderResponseError("weather response has an invalid 'daily' block") from exc
        codes = daily.weather_code or [None] * len(daily.time)
        days = [
            DailyWeather(day=d, temp_max_c=hi, temp_min_c=lo, precipitation_mm=rain, weather_code=code)
            for d, hi, lo, rain, code in zip(
                daily.time, daily.temperature_2m_max, daily.temperature_2m_min, daily.precipitation_sum, codes,
                strict=True,
            )
            if hi is not None and lo is not None and rain is not None  # skip days with gaps
        ]
        if not days:
            raise NoDataError("the weather service has no data for that location and period")
        return days


# --- Date parsing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    granularity: Literal["day", "month"]


def parse_period(value: str, today: date) -> Period:
    """Parse 'YYYY-MM-DD', 'YYYY-MM', 'November', 'Nov 2026' or 'November 2026'.

    A month name without a year means its next occurrence (this year if not yet over).
    Raises ValueError with a message listing accepted formats.
    """
    text = (value or "").strip()
    accepted = "Use YYYY-MM-DD, YYYY-MM, or a month name such as 'November' or 'November 2026'."
    if not text:
        raise ValueError(f"date_or_month is empty. {accepted}")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            day = date.fromisoformat(text)
            period = Period(day, day, "day")
        elif match := re.fullmatch(r"(\d{4})-(\d{2})", text):
            period = _month_period(int(match[1]), int(match[2]))
        elif match := re.fullmatch(r"([A-Za-z]+)\.?(?:\s+(\d{4}))?", text):
            month = _MONTHS.get(match[1].lower())
            if month is None:
                raise ValueError(f"unrecognised date {text!r}. {accepted}")
            year = int(match[2]) if match[2] else today.year + (1 if month < today.month else 0)
            period = _month_period(year, month)
        else:
            raise ValueError(f"unrecognised date {text!r}. {accepted}")
    except ValueError as exc:
        if "unrecognised" in str(exc):
            raise
        raise ValueError(f"invalid date {text!r}: {exc}. {accepted}") from exc
    if not EARLIEST_YEAR <= period.start.year <= today.year + MAX_FUTURE_YEARS:
        raise ValueError(
            f"year {period.start.year} is outside the supported range {EARLIEST_YEAR}-{today.year + MAX_FUTURE_YEARS}"
        )
    return period


def _month_period(year: int, month: int) -> Period:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    last = calendar.monthrange(year, month)[1]
    return Period(date(year, month, 1), date(year, month, last), "month")


# --- Summarising --------------------------------------------------------------------------------


def _temperature_word(avg_high: float) -> str:
    for limit, word in ((5, "cold"), (15, "cool"), (25, "mild"), (30, "warm")):
        if avg_high <= limit:
            return word
    return "hot"


def _conditions(days: list[DailyWeather], avg_high: float, rainy_days: int) -> str:
    temp = _temperature_word(avg_high)
    if len(days) == 1 and days[0].weather_code in _WMO_CODES:
        return f"{temp}, {_WMO_CODES[days[0].weather_code]}"
    share = rainy_days / len(days)
    wetness = "mostly dry" if share < 0.1 else "some wet days" if share < 0.35 else "frequent precipitation"
    return f"{temp}, {wetness}"


def _summarise(days: list[DailyWeather]) -> dict:
    avg_high = sum(d.temp_max_c for d in days) / len(days)
    avg_low = sum(d.temp_min_c for d in days) / len(days)
    rainy = sum(1 for d in days if d.precipitation_mm >= RAIN_DAY_MM)
    return {
        "conditions": _conditions(days, avg_high, rainy),
        "temp_range_c": [min(d.temp_min_c for d in days), max(d.temp_max_c for d in days)],
        "avg_high_c": round(avg_high, 1),
        "avg_low_c": round(avg_low, 1),
        "total_precipitation_mm": round(sum(d.precipitation_mm for d in days), 1),
        "rainy_days": rainy,
        "days_with_data": len(days),
    }


# --- Public tool function -----------------------------------------------------------------------


def get_weather_forecast(
    city: str, date_or_month: str, provider: WeatherProvider, today: date | None = None
) -> dict:
    """Return weather for `city` on a date or in a month as a JSON-serialisable dict.

    Expected failures (bad input, unknown city, provider errors) are returned with
    status="error" and a message; no weather values are ever invented.
    """
    today = today or date.today()
    clean_city = (city or "").strip()
    base = {"city": clean_city, "source": provider.name, "is_live": provider.is_live}

    def error(error_type: str, message: str) -> dict:
        return WeatherResult(status="error", error_type=error_type, message=message, **base).model_dump(
            mode="json", exclude_none=True
        )

    if not clean_city or len(clean_city) > 100 or not _CITY_PATTERN.match(clean_city):
        return error("invalid_city", "city must be a place name such as 'Tokyo' (letters only, max 100 chars)")
    try:
        period = parse_period(date_or_month, today)
    except ValueError as exc:
        return error("invalid_date", str(exc))

    try:
        location = provider.geocode(clean_city.split(",")[0].strip())
        data_kind, days, window, reference, limitations = _fetch(provider, location, period, today)
    except WeatherProviderError as exc:
        return error(exc.error_type, str(exc))

    requested_days = (window[1] - window[0]).days + 1
    if len(days) < requested_days:
        limitations.append(f"Data was available for only {len(days)} of {requested_days} days in the period.")
    return WeatherResult(
        status="ok",
        data_kind=data_kind,
        location=location,
        period_start=period.start,
        period_end=period.end,
        reference_period=reference,
        limitations=limitations,
        **base,
        **_summarise(days),
    ).model_dump(mode="json", exclude_none=True)


def _fetch(provider, location, period, today):
    """Pick the right endpoint/window for the period; returns (kind, days, window, reference, limits)."""
    archive_cutoff = today - timedelta(days=ARCHIVE_LAG_DAYS)
    forecast_limit = today + timedelta(days=FORECAST_MAX_DAYS_AHEAD)
    limitations: list[str] = []

    if period.end <= archive_cutoff:
        days = provider.daily(location, period.start, period.end, "archive")
        return "observed_historical", days, (period.start, period.end), None, limitations

    if period.granularity == "day" and period.start <= forecast_limit:
        days = provider.daily(location, period.start, period.end, "forecast")
        return "forecast", days, (period.start, period.end), None, limitations

    # Month-only, or a day too far ahead for a forecast: use last year's observations as a guide.
    start, end = period.start, period.end
    if period.granularity == "day":
        start, end = start - timedelta(days=CLIMATE_WINDOW_DAYS), end + timedelta(days=CLIMATE_WINDOW_DAYS)
    year = start.year - 1
    while _shift_year(end, year) > archive_cutoff:
        year -= 1
    ref_start, ref_end = _shift_year(start, year), _shift_year(end, year)
    days = provider.daily(location, ref_start, ref_end, "archive")
    reference = f"{ref_start.isoformat()} to {ref_end.isoformat()} (observed)"
    if period.granularity == "month":
        limitations.append(
            "Month-only request: no day-by-day forecast exists this far ahead. These figures summarise "
            f"observed weather for the same month in {year}. Ask for a specific date within the next "
            f"{FORECAST_MAX_DAYS_AHEAD + 1} days for a real forecast."
        )
    else:
        limitations.append(
            f"Date is beyond the {FORECAST_MAX_DAYS_AHEAD + 1}-day forecast range. These figures summarise "
            f"observed weather around the same date in {year}; they are a guide, not a forecast."
        )
    return "climate_reference", days, (ref_start, ref_end), reference, limitations


def _shift_year(day: date, year: int) -> date:
    try:
        return day.replace(year=year)
    except ValueError:  # 29 February in a non-leap year
        return day.replace(year=year, day=28)
