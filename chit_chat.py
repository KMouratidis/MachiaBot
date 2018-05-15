from collections import defaultdict
from skills import get_weather
import re
from numpy import random
from itertools import chain

# TODO: responses that need to call some API and get back info, then format into strings
api_responses = {
    'weather_today': "Today the weather in {} is {}.",
}


# TODO: responses that need to do some 'thinking'
thinking_responses = {
    'weather': "Today's weather is {}",  # nice, bad, etc, get info from get_weather, judge it, then decide
}


# TODO: Add more lookup patterns
# Write patterns in a dict that will be used both for regexps lookup patterns and for responses
lookup_patterns = {
    "greetings": "hello|hey|hi|yo|greetings",
    "partings": "goodbye|good bye|adios|bye|see you|see ya",
    "thanks": "thank you|thanks|arigato|grazie|thx|thnks",

    "NER": "[A-Z]{1}[a-z]+",  # + is maybe wrong, but this approach is too crude either way

}


# These templates are to be used as responses and
# can handle both hardcoded values and formatted text
response_templates = defaultdict(list)

for key, string in lookup_patterns.items():
    response_templates[key] = string.split("|")

# TODO: Add more hardcoded response templates
response_templates["greetings_funny"] = (["{} to you too!", "{}, kiddo!", "I don't want to talk now, go away!",
                                          "'{}' my ass! Greet me properly!", "{}!!!!", "Hell-o ;)", "{} earthling!",
                                          
                                          ])





if __name__ == "__main__":
    city, forecast = get_weather()
    print(api_responses['weather_today'].format(city, forecast.lower()))

    string = "Hello there, I am Kostas! Thanks for the mention. Now bye!"
    for key in lookup_patterns.keys():
        print(re.findall(lookup_patterns[key], string))

    sample_greeting = re.findall(lookup_patterns['greetings'], string.lower())[0]
    if (sample_greeting is not None) and (sample_greeting in lookup_patterns["greetings"]):
        print(sample_greeting.capitalize(), "to you too!") # reply in kind

        random_template = random.choice(response_templates["greetings_funny"])
        print(random_template.format(sample_greeting), end="\n\n")  # reply in kind with variation

        # list(chain(iters)) concatenates the lists
        random_template = random.choice(list(chain(response_templates["greetings_funny"],
                                                   response_templates["greetings"])))
        print(random_template.format(sample_greeting), end="\n\n")  # reply in kind with variation

        for key, value in response_templates.items():
            print("All ({}) possible responses in the category '{}': {}".format(len(value), key, value))