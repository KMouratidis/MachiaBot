from weather import Weather, Unit

def get_weather(location="Thessaloniki", *, units=Unit.CELSIUS):
    """

    A function that takes a location as input and returns the weather by quering the Yahoo API.
    Location can be given either by string, or integer (see WOEID via http://weather.yahoo.com)
    or tuple containing 2 numbers (latitude, longitude)

    :param location: Either string, or int, or tuple
    :param units: Unit.CELSIUS or Unit.FAHRENHEIT
    :return: location, condition.text
    """

    weather = Weather(unit=units)

    if isinstance(location, str):
        location = location.lower()
        lookup = weather.lookup_by_location(location)
    elif isinstance(location, int):
        lookup = weather.lookup(location)
    elif isinstance(location, tuple):
        lat, long = float(location[0]), float(location[1])
        lookup = weather.lookup_by_latlng(lat, long)
    else:
        raise NotImplementedError

    weather_condition = lookup.condition
    city = lookup.location.city # if country is wanted: lookup.location.country

    return city, weather_condition.text



if __name__ == "__main__":
    print(get_weather("Athens"))