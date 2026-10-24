from pydantic import BaseModel, Field


class AirQualityInput(BaseModel):
    pm2_5: float = Field(..., ge=0.0, le=1000.0, description="PM2.5 concentration in ug/m3")
    pm10: float = Field(..., ge=0.0, le=1500.0, description="PM10 concentration in ug/m3")
    no2: float = Field(..., ge=0.0, le=1000.0, description="NO2 concentration in ug/m3")
    so2: float = Field(..., ge=0.0, le=1000.0, description="SO2 concentration in ug/m3")
    co: float = Field(..., ge=0.0, le=100.0, description="CO concentration in mg/m3")
    temperature: float = Field(..., ge=-20.0, le=60.0, description="Ambient temperature in Celsius")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity in percentage")
    wind_speed: float = Field(..., ge=0.0, le=150.0, description="Wind speed in km/h or m/s")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pm2_5": 42.5,
                "pm10": 85.0,
                "no2": 24.0,
                "so2": 12.0,
                "co": 0.8,
                "temperature": 28.5,
                "humidity": 55.0,
                "wind_speed": 3.2
            }
        }
    }


class PredictionOutput(BaseModel):
    predicted_aqi: float
    category: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool