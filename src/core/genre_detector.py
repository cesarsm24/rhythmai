class GenreDetector:

    GENRES_MAP = {
        "reggaeton": "reggaeton",
        "reguetón": "reggaeton",
        "trap": "trap",
        "techno": "techno",
        "tecno": "techno",
        "house": "house",
        "edm": "edm",
        "electrónica": "electronic",
        "electronica": "electronic",
        "electro": "electronic",
        "dance": "dance",
        "fiesta": "dance",
        "party": "dance",
        "rock": "rock",
        "metal": "metal",
        "pop": "pop",
        "indie": "indie-pop",
        "latino": "latin",
        "latina": "latin",
        "salsa": "latin",
        "bachata": "latin",
        "rap": "hip-hop",
        "hiphop": "hip-hop",
        "hip-hop": "hip-hop"
    }

    @staticmethod
    def detect(text: str):
        text = text.lower()
        found = []

        for word, genre in GenreDetector.GENRES_MAP.items():
            if word in text:
                found.append(genre)

        return found[:2]  # máximo 2 como Spotify permite
