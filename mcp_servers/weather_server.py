"""MCP 天气 server - 基于 Open-Meteo 免费 API，无需 API key。

通过 stdio 传输，提供实时天气查询（支持中文城市名）。

启动方式：
    venv/bin/python mcp_servers/weather_server.py

CLI 接入：
    agent chat --mcp-server "stdio:weather:venv/bin/python:mcp_servers/weather_server.py"
"""
import json
import sys

import httpx
from mcp.server.mcpserver import MCPServer

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "晴", 1: "大部晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "霰",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _geocode(city: str) -> dict | None:
    resp = httpx.get(GEOCODE_URL, params={"name": city, "count": 1, "language": "zh"}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results")
    return results[0] if results else None


def _fetch_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,wind_direction_10m,pressure_msl",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 3,
    }
    resp = httpx.get(FORECAST_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


mcp = MCPServer("weather-server", version="0.1.0")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的实时天气（支持中文城市名，如"北京"、"杭州"、"上海"）。

    Args:
        city: 城市名称（中英文均可）
    """
    try:
        geo = _geocode(city)
        if not geo:
            return f"未找到城市 '{city}'"

        data = _fetch_weather(geo["latitude"], geo["longitude"])
        cur = data["current"]
        code = cur["weather_code"]
        desc = WEATHER_CODES.get(code, f"未知({code})")
        daily = data["daily"]

        lines = [
            f"📍 {geo.get('name', city)}（{geo.get('country', '')}）",
            f"🌡️ 当前温度: {cur['temperature_2m']}°C（体感 {cur['apparent_temperature']}°C）",
            f"🌤️ 天气: {desc}",
            f"💧 湿度: {cur['relative_humidity_2m']}%",
            f"💨 风速: {cur['wind_speed_10m']} km/h",
            f"📊 气压: {cur['pressure_msl']} hPa",
            "",
            "未来 3 天预报:",
        ]
        for i, date in enumerate(daily["time"]):
            d_code = daily["weather_code"][i]
            d_desc = WEATHER_CODES.get(d_code, f"未知({d_code})")
            lines.append(
                f"  {date}: {d_desc}，"
                f"{daily['temperature_2m_min'][i]}~{daily['temperature_2m_max'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]}mm"
            )

        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"天气查询失败（网络错误）: {e}"
    except (KeyError, IndexError) as e:
        return f"天气查询失败（数据解析错误）: {e}"


if __name__ == "__main__":
    mcp.run("stdio")
