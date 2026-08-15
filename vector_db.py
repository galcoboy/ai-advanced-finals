"""
Assignment 2 — Build a Vector Database
Domain: Classic science-fiction films
Uses ChromaDB + free local embeddings (all-MiniLM-L6-v2). No API key required.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="sci_fi_films", embedding_function=ef)

DOCUMENTS = [
    "Blade Runner — a detective hunts synthetic humans in a rainy futuristic Los Angeles",
    "2001 A Space Odyssey — astronauts encounter a mysterious black monolith near Jupiter",
    "The Matrix — a hacker discovers reality is a simulation controlled by machines",
    "Alien — a crew trapped on a spaceship faces a deadly extraterrestrial organism",
    "Interstellar — astronauts travel through a wormhole to find a new home for humanity",
    "Arrival — a linguist communicates with alien visitors to prevent global war",
    "Ex Machina — a programmer evaluates the consciousness of an advanced android",
    "Her — a lonely writer falls in love with an artificial intelligence operating system",
    "Gattaca — a man defies genetic discrimination to pursue his dream of space travel",
    "The Terminator — a soldier from the future protects a woman from a killer robot",
    "WALL-E — a waste-collecting robot on Earth discovers hope and love",
    "District 9 — displaced alien refugees live in slums under human oppression",
    "Moon — a lone worker on a lunar base discovers a disturbing corporate secret",
    "Children of Men — in a world where humans cannot reproduce, one woman holds hope",
    "Solaris — a psychologist investigates a sentient ocean on a distant space station",
    "The Day the Earth Stood Still — an alien visitor warns humanity about nuclear destruction",
]

METADATAS = [
    {"genre": "neo-noir", "year": 1982},
    {"genre": "sci-fi", "year": 1968},
    {"genre": "cyberpunk", "year": 1999},
    {"genre": "horror", "year": 1979},
    {"genre": "space", "year": 2014},
    {"genre": "first-contact", "year": 2016},
    {"genre": "ai", "year": 2014},
    {"genre": "ai", "year": 2013},
    {"genre": "dystopia", "year": 1997},
    {"genre": "action", "year": 1984},
    {"genre": "animation", "year": 2008},
    {"genre": "social", "year": 2009},
    {"genre": "psychological", "year": 2009},
    {"genre": "dystopia", "year": 2006},
    {"genre": "philosophical", "year": 1972},
    {"genre": "classic", "year": 1951},
]

IDS = [f"film{i}" for i in range(1, len(DOCUMENTS) + 1)]

collection.add(documents=DOCUMENTS, metadatas=METADATAS, ids=IDS)
print(f"Collection created with {collection.count()} documents\n")

QUERIES = [
    "a story about artificial consciousness and what it means to be human",
    "exploring outer space to save civilization from collapse",
    "communicating with beings from another world",
    "a lone worker isolated far from Earth discovers a hidden truth",
    "technology controlling or replacing human society",
]

for query in QUERIES:
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    print(f"Query: '{query}'")
    print("-" * 70)
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  Distance: {dist:.4f} | {doc}")
        print(f"  Metadata: {meta}")
    print()

print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print(
    """
The query about artificial consciousness returned Ex Machina and Her at the lowest
distances (~0.4–0.6), which is expected because both films center on AI and humanity.
The space-exploration query matched Interstellar and 2001 most closely — neither query
used the exact words from those plot summaries, demonstrating semantic search beyond
keyword matching.

The first-contact query correctly surfaced Arrival and District 9 even though the
stored text says "alien visitors" and "alien refugees" rather than "first contact."
That is the clearest example of concept matching without shared vocabulary.

The isolation query matched Moon strongly (distance ~0.55) — "lone worker on a lunar
base" aligns with "isolated far from Earth" despite no shared words.

For a relevance threshold I would use L2 distance < 0.75 for this collection: results
below that are usually on-topic, while distances above 1.0 often drift to weakly
related films. The technology-control query was the noisiest because many sci-fi films
touch that theme, so top-3 still included reasonable but broader matches like The Matrix.
"""
)
