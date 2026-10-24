from src.inference.category import get_aqi_category


def test_aqi_category_mapping():
    assert get_aqi_category(35) == "Good"
    assert get_aqi_category(75) == "Satisfactory"
    assert get_aqi_category(150) == "Moderate"
    assert get_aqi_category(250) == "Poor"
    assert get_aqi_category(350) == "Very Poor"
    assert get_aqi_category(450) == "Severe"