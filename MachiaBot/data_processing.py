import re

# total words: ~60k

def process_data(file="Dialogues in Hell.txt"):
    with open(file, "r") as f:
        rd = f.read()

    # remove chapter headings
    rd = re.sub("\w+ Dialogue", " ", rd)
    # substitute 3 or more newlines to 2; won't matter later, but it's okay for now
    rd = re.sub("\n{3,}", "\n\n", rd)

    # split text based on who speaks
    splitted_text = re.split("\n(Machiavelli|Montesquieu):", rd)

    # the list should have the form: [speaker, text, speaker, text, ...]
    speaker = splitted_text[1::2]
    dialogue = [re.sub("\n", "", x.strip()) for x in splitted_text[2::2]] # also removes \n and whitespace

    return list(zip(speaker, dialogue))

if __name__ == "__main__":
    process_data()

# for i in range(15):
#     print("Speaker: ", speaker[i], "\nDialogue: ", dialogue[i])
#     print("\n\n")
#
# print(len(speaker)) # 873 replies