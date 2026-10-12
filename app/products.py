"""The products the system has documentation for: friendly names for the UI, plus a few example claims each.

The product_id values must match the folder names in data/raw (tests/test_api.py checks this).
The example claims come from the development eval set, so the UI notes that they are not held-out evidence.
"""

PRODUCT_CATALOG = [
    {
        "product_id": "alps_helix_2",
        "name": "ALPS Mountaineering Helix 2-Person Tent",
        "category": "Outdoor gear",
        "examples": [
            "sleeps 2 people",
            "the rainfly has a 500mm waterproof coating",
            "comes with a lifetime warranty",
        ],
    },
    {
        "product_id": "black_diamond_spot_400",
        "name": "Black Diamond Spot 400 Headlamp",
        "category": "Outdoor gear",
        "examples": [
            "waterproof to 1 meter for 30 minutes",
            "operates for 100 hours on high brightness",
            "includes a carrying case",
        ],
    },
    {
        "product_id": "nalgene_32oz_wide_mouth",
        "name": "Nalgene 32 oz Wide Mouth Bottle",
        "category": "Outdoor gear",
        "examples": [
            "BPA-free",
            "not safe to put in the dishwasher",
            "includes a built-in compass",
        ],
    },
    {
        "product_id": "jbl_flip_6",
        "name": "JBL Flip 6 Speaker",
        "category": "Electronics",
        "examples": [
            "IP67 waterproof and dustproof",
            "24 hours of battery life",
            "has a built-in flashlight",
        ],
    },
    {
        "product_id": "anker_313_powercore_10k",
        "name": "Anker 313 Power Bank (PowerCore 10K)",
        "category": "Electronics",
        "examples": [
            "10000mAh capacity",
            "20000mAh capacity",
            "supports wireless charging",
        ],
    },
    {
        "product_id": "soundcore_life_q20",
        "name": "Soundcore Life Q20 Headphones",
        "category": "Electronics",
        "examples": [
            "reduces ambient noise by up to 90%",
            "reduces ambient noise by up to 50%",
            "is waterproof",
        ],
    },
    {
        "product_id": "instant_pot_duo",
        "name": "Instant Pot Duo 7-in-1",
        "category": "Kitchen",
        "examples": [
            "has a 6-quart capacity",
            "has a 10-quart capacity",
            "is compatible with induction cooktops",
        ],
    },
    {
        "product_id": "philips_airfryer_hd9252",
        "name": "Philips 3000 Series Airfryer L (HD9252)",
        "category": "Kitchen",
        "examples": [
            "has a 4.1 liter capacity",
            "has an 8 liter capacity",
            "includes a physical recipe book",
        ],
    },
    {
        "product_id": "ninja_blender_bl621",
        "name": "Ninja Professional Blender (BL621)",
        "category": "Kitchen",
        "examples": [
            "has a 1100-watt motor",
            "has a 500-watt motor",
            "comes with a travel strap",
        ],
    },
]

PRODUCT_IDS = {p["product_id"] for p in PRODUCT_CATALOG}
