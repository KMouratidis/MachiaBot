import re

# total words: ~60k
"""
Did some manual cleaning on the txt file:
    1) Changed every 'Montequieu' to Montesquieu
    2) Changed every 'Montesquieu:' that was not the beginning of a sentence to 'Montesquieu,'
"""

def process_data(file="Dialogues in Hell.txt"):
    with open(file, "r") as f:
        rd = f.read()

    # remove chapter headings
    rd = re.sub("\w+ Dialogue", " ", rd)
    # substitute 3 or more newlines to 2; won't matter later, but it's okay for now
    rd = re.sub("\n{3,}", "\n\n", rd)
    # remove references like: [2]
    rd = re.sub(r"\[\d+\]", "", rd)

    # split text based on who speaks
    splitted_text = re.split("(Machiavelli|Montesquieu):", rd)

    # the list should have the form: [speaker, text, speaker, text, ...]
    speaker = splitted_text[1::2]
    dialogue = [re.sub("\n", "", x.strip()) for x in splitted_text[2::2]] # also removes \n and whitespace

    return list(zip(speaker, dialogue))

if __name__ == "__main__":
    data = process_data()

    for speaker, dialogue in data[:5]:
        print("{}: {}".format(speaker, dialogue))

    print("Total dialogues:", len(data)) # 873 replies