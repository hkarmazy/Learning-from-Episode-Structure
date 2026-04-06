"""CIFAR-100 class selection for Attractor Landscape Training.

32 classes selected from CIFAR-100, ordered by Zipf rank.
Rank 0 (baby) is the most frequent; rank 31 (fox) is the rarest.
Ranks 24-31 are designated as 'rare' classes.
"""

# CIFAR-100 fine-label indices in Zipf rank order
CIFAR100_CLASS_INDICES = [
    2,   # baby
    28,  # cup
    9,   # bottle
    13,  # bus
    58,  # pickup_truck
    5,   # bed
    22,  # clock
    84,  # table
    61,  # plate
    1,   # aquarium_fish
    3,   # bear
    65,  # rabbit
    50,  # mouse
    6,   # bee
    27,  # crocodile
    93,  # turtle
    99,  # worm
    53,  # orange
    51,  # mushroom
    59,  # pine_tree
    0,   # apple
    86,  # telephone
    31,  # elephant
    79,  # spider
    43,  # lion          (rare)
    88,  # tiger         (rare)
    14,  # butterfly     (rare)
    8,   # bicycle       (rare)
    69,  # rocket        (rare)
    90,  # train         (rare)
    70,  # rose          (rare)
    34,  # fox           (rare)
]

CIFAR100_CLASS_NAMES = [
    'baby', 'cup', 'bottle', 'bus', 'pickup_truck',
    'bed', 'clock', 'table', 'plate', 'aquarium_fish',
    'bear', 'rabbit', 'mouse', 'bee', 'crocodile',
    'turtle', 'worm', 'orange', 'mushroom', 'pine_tree',
    'apple', 'telephone', 'elephant', 'spider',
    'lion', 'tiger', 'butterfly', 'bicycle',
    'rocket', 'train', 'rose', 'fox',
]

N_CLASSES = 32
RARE_CLASSES = list(range(24, 32))  # lion, tiger, butterfly, bicycle, rocket, train, rose, fox
COMMON_CLASSES = list(range(24))

# Reverse mapping: CIFAR-100 fine-label index → our rank (0-31)
CIFAR_IDX_TO_RANK = {idx: rank for rank, idx in enumerate(CIFAR100_CLASS_INDICES)}
