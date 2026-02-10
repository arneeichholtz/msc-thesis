# ---------------- IPA and A course in Phonetics ----------------
# This section defines phonological features based on the International Phonetic Alphabet (IPA) and "A Course in Phonetics" by Peter Ladefoged.
# 
# - Affricate was added to the manner labels.
# - Labio-velar is added as a place of articulation, in IPA it is not strictly part of the table, though it is mentioned below the table.
# - In the book, glottal is under "Manner" instead of "Place". But in IPA, glottal is a place of articulation.


# ---------------------
# Consonants
# ---------------------
manner_labels = ["Stop", "Nasal", "Fricative", "Approximant", "Affricate", "Tap", "Lateral Approximant"]
place_labels = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Post-alveolar", "Palatal", "Velar", "Glottal", "Labio-velar"]
voice_labels = ["Voiced", "Voiceless"]

consonants_labels = {
    # [IPA, Manner, Place, Voice]
    "p":  ["p", "Stop", "Bilabial", "Voiceless"],
    "b":  ["b", "Stop", "Bilabial", "Voiced"],
    "t":  ["t", "Stop", "Alveolar", "Voiceless"],
    "d":  ["d", "Stop", "Alveolar", "Voiced"],
    "k":  ["k", "Stop", "Velar", "Voiceless"],
    "g":  ["g", "Stop", "Velar", "Voiced"],
    "dx": ["ɾ", "Tap", "Alveolar", "Voiced"],
    "ch": ["tʃ", "Affricate", "Post-alveolar", "Voiceless"],
    "jh": ["dʒ", "Affricate", "Post-alveolar", "Voiced"],
    "f":  ["f", "Fricative", "Labiodental", "Voiceless"],
    "v":  ["v", "Fricative", "Labiodental", "Voiced"],
    "th": ["θ", "Fricative", "Dental", "Voiceless"],
    "dh": ["ð", "Fricative", "Dental", "Voiced"],
    "s":  ["s", "Fricative", "Alveolar", "Voiceless"],
    "z":  ["z", "Fricative", "Alveolar", "Voiced"],
    "sh": ["ʃ", "Fricative", "Post-alveolar", "Voiceless"],
    "hh": ["h", "Fricative", "Glottal", "Voiceless"],
    "m":  ["m", "Nasal", "Bilabial", "Voiced"],
    "n":  ["n", "Nasal", "Alveolar", "Voiced"],
    "ng": ["ŋ", "Nasal", "Velar", "Voiced"],
    "l":  ["l", "Lateral Approximant", "Alveolar", "Voiced"],           # "l": ["Approximant", "Alveolar", "Voiced"] -- difference Lateral/Approximant valuable?
    "r":  ["ɹ", "Approximant", "Alveolar", "Voiced"],                   # "r": ["Approximant", "Post-alveolar", "Voiced"] -- English "red"   
    "w":  ["w", "Approximant", "Labio-velar", "Voiced"],
    "y":  ["j", "Approximant", "Palatal", "Voiced"]
}

# Create mapping dictionaries and construct consonants IDs dict
manner_map = {label: i for i, label in enumerate(manner_labels)}
place_map = {label: i for i, label in enumerate(place_labels)}
voice_map = {label: i for i, label in enumerate(voice_labels)}

consonants_ids = {
    phoneme: [
        manner_map[features[1]], 
        place_map[features[2]], 
        voice_map[features[3]]
    ]
    for phoneme, features in consonants_labels.items()
}
# E.g. phoneme "g" --> [Stop, Velar, Voiced] = [0, 6, 0]



# ---------------------
# Vowels
# ---------------------
tongue_height_labels = ["Close", "Near-close", "Mid", "Open-mid", "Near-open", "Open"]
tongue_backness_labels = ["Front", "Near-front", "Central", "Near-back", "Back"]
lip_roundedness_labels = ["Rounded", "Unrounded"]

vowels_labels = {
    # [IPA, Height, Backness, Roundedness]
    "iy": ["i", "Close", "Front", "Unrounded"],
    "ih": ["ɪ", "Near-close", "Near-front", "Unrounded"],
    "eh": ["ɛ", "Open-mid", "Front", "Unrounded"],
    "ae": ["æ", "Near-open", "Front", "Unrounded"],
    "aa": ["ɑ", "Open", "Back", "Unrounded"],
    "ah": ["ʌ", "Open-mid", "Back", "Unrounded"],           # "ah": ["ʌ", "Open-mid", "Central", "Unrounded"], -- American/British speakers have Central rather than Back (from Ladefoged)
    "uh": ["ʊ", "Near-close", "Near-back", "Rounded"],
    "uw": ["u", "Close", "Back", "Rounded"],
    "er": ["ɝ", "Open-mid", "Central", "Rounded"]          # "er": ["ɝ", "Mid", "Central", "Rounded"] -- Mid-central vowel (Ladefoged) 
                                                           # This vowel does not fit on the chart because it cannot be described simply 
                                                           # in terms of the features high–low, front–back, and rounded–unrounded.
}

# Create mapping dictionaries and construct vowels IDs dict
height_map = {label: i for i, label in enumerate(tongue_height_labels)}
backness_map = {label: i for i, label in enumerate(tongue_backness_labels)}
roundedness_map = {label: i for i, label in enumerate(lip_roundedness_labels)}

vowels_ids = {
    phoneme: [
        height_map[features[1]], 
        backness_map[features[2]], 
        roundedness_map[features[3]]
    ]
    for phoneme, features in vowels_labels.items()
}
# E.g. phoneme "iy" --> [Close, Front, Unrounded] = [0, 0, 1]


# ---------------------
# Diphthongs
# ---------------------
# This is a simplified representation of diphthongs focusing on their starting and ending vowel features, based on the book.
diphthongs = {
    # ARPAbet: [IPA, Start_Feature, End_Feature]
    "ey": ["eɪ", "Mid/Near-front", "Close/Front"],
    "ay": ["aɪ", "Open/Central", "Close/Front"],
    "oy": ["ɔɪ", "Open-mid/Back", "Close/Front"],
    "aw": ["aʊ", "Open/Central", "Close/Back"],
    "ow": ["oʊ", "Mid/Back", "Close/Back"]
}

# ----------------  ----------------