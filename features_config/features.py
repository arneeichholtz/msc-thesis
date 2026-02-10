

# ---------------- King & Taylor paper ----------------

# Phonological features, both their label used in the paper and label in natural language
phonation_labels = {"v": "Voiced", "uv": "Unvoiced", "s": "silence"}
manner_labels    = {"v": "Vowel", "o": "Occlusive", "a": "Approximant", "n": "Nasal", "f": "Fricative", "s": "silence"}
place_labels     = {"lo": "Low", "m": "Mid", "h": "High", "l": "Labial", "c": "Coronal", "d": "Dental", "v": "Velar", "g": "Glottal", "s": "silence"}
fb_labels        = {"b": "Back", "f": "Front", "n": "Neutral", "s": "silence"}
round_labels     = {"u": "Unrounded", "r": "Rounded", "s": "silence"}
central_labels   = {"c": "Central", "f": "Full", "n": "nil", "s": "silence"}

# Data extracted from the King & Taylor Appendix (MV System) 
mv_data = {
    # Phoneme: [phonation, manner, place, front-back, roundness, centrality]
    "aa":   ["v", "v", "lo", "b", "u", "c"],
    "ae":   ["v", "v", "lo", "f", "u", "f"],
    "ah":   ["v", "v", "lo", "f", "u", "f"],
    "ao":   ["v", "v", "m", "b", "r", "f"],
    "aw":   ["v", "v", "m", "b", "r", "f"],
    "ax":   ["v", "v", "m", "n", "u", "c"],
    "ax-h": ["v", "v", "m", "n", "u", "c"],
    "axr":  ["v", "v", "m", "n", "u", "c"],
    "ay":   ["v", "v", "m", "f", "u", "f"],
    "b":    ["v", "o", "l", "f", "u", "n"],
    "bcl":  ["uv", "o", "l", "f", "u", "n"],
    "ch":   ["uv", "f", "c", "f", "u", "n"],
    "d":    ["v", "o", "c", "f", "u", "n"],
    "dcl":  ["uv", "o", "c", "f", "u", "n"],
    "dh":   ["v", "f", "d", "f", "u", "n"],
    "dx":   ["v", "o", "c", "f", "u", "n"],
    "eh":   ["v", "v", "m", "f", "u", "f"],
    "el":   ["v", "a", "c", "f", "u", "f"],
    "em":   ["v", "n", "l", "f", "u", "n"],
    "en":   ["v", "n", "c", "f", "u", "n"],
    "eng":  ["v", "n", "v", "b", "u", "n"],
    "er":   ["v", "a", "v", "b", "u", "f"],
    "ey":   ["v", "v", "h", "f", "u", "f"],
    "f":    ["uv", "f", "d", "f", "u", "n"],
    "g":    ["v", "o", "v", "b", "u", "n"],
    "gcl":  ["uv", "o", "v", "b", "u", "n"],
    "hh":   ["uv", "f", "g", "b", "u", "n"],
    "hv":   ["v", "f", "g", "b", "u", "n"],
    "ih":   ["v", "v", "h", "f", "u", "f"],
    "ix":   ["v", "v", "h", "f", "u", "f"],
    "iy":   ["v", "v", "h", "f", "u", "f"],
    "jh":   ["v", "f", "c", "f", "u", "n"],
    "k":    ["uv", "o", "v", "b", "u", "n"],
    "kcl":  ["uv", "o", "v", "b", "u", "n"],
    "l":    ["v", "a", "c", "f", "u", "n"],
    "m":    ["v", "n", "l", "f", "u", "n"],
    "n":    ["v", "n", "c", "f", "u", "n"],
    "ng":   ["v", "n", "v", "b", "u", "n"],
    "nx":   ["v", "n", "c", "f", "u", "n"],
    "ow":   ["v", "v", "h", "b", "r", "f"],
    "oy":   ["v", "v", "h", "b", "r", "f"],
    "p":    ["uv", "o", "l", "f", "u", "n"],
    "pcl":  ["uv", "o", "l", "f", "u", "n"],
    "q":    ["uv", "o", "g", "b", "u", "n"],
    "r":    ["v", "a", "v", "b", "u", "n"],
    "s":    ["uv", "f", "c", "f", "u", "n"],
    "sh":   ["uv", "f", "c", "f", "u", "n"],
    "t":    ["uv", "o", "c", "f", "u", "n"],
    "tcl":  ["uv", "o", "c", "f", "u", "n"],
    "th":   ["v", "f", "d", "f", "u", "n"],
    "uh":   ["v", "v", "h", "b", "r", "f"],
    "uw":   ["v", "v", "h", "b", "r", "f"],
    "ux":   ["v", "v", "h", "b", "r", "f"],
    "v":    ["v", "f", "d", "f", "u", "n"],
    "w":    ["v", "a", "l", "f", "r", "n"],
    "y":    ["v", "a", "v", "b", "u", "n"],
    "z":    ["v", "f", "c", "f", "u", "n"],
    "zh":   ["v", "f", "c", "f", "u", "n"],
    "sil":  ["s", "s", "s", "s", "s", "s"] 
}

phoneme_mapping = {
    'p': 'p',
    'b': 'b',
    't': 't',
    'd': 'd',
    'k': 'k',
    'g': 'g',
    'dx': 'dx',
    'f': 'f',
    'v': 'v',
    'dh': 'dh',
    'th': 'th',
    's': 's',
    'z': 'z',
    'r': 'r',
    'q': 'q',
    'w': 'w',
    'y': 'y',
    'jh': 'jh',
    'ch': 'ch',
    'iy': 'iy',
    'eh': 'eh',
    'ey': 'ey',
    'ae': 'ae',
    'aw': 'aw',
    'ay': 'ay',
    'oy': 'oy',
    'ow': 'ow',
    'uh': 'uh',
    'ah': 'ah',
    'ax': 'ah',
    'ax-h': 'ah',
    'aa': 'aa',
    'ao': 'aa',
    'er': 'er',
    'axr': 'er',
    'hh': 'hh',
    'hv': 'hh',
    'ih': 'ih',
    'ix': 'ih',
    'l': 'l',
    'el': 'l',
    'm': 'm',
    'em': 'm',
    'n': 'n',
    'en': 'n',
    'nx': 'n',
    'ng': 'ng',
    'eng': 'ng',
    'sh': 'sh',
    'zh': 'sh',
    'uw': 'uw',
    'ux': 'uw',
    'pcl': 'sil',
    'bcl': 'sil',
    'tcl': 'sil',
    'dcl': 'sil',
    'kcl': 'sil',
    'gcl': 'sil',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',
    'sil': 'sil'
}


# ----------------  ----------------










