"""Target catalogue: real platform formats, as hidden scoring targets will be."""

CATALOGUE = [
    ('Instagram Post', 1080, 1080), ('Instagram Portrait', 1080, 1350), ('Instagram Story', 1080, 1920),
    ('YouTube Thumbnail', 1280, 720), ('Facebook Link', 1200, 628), ('X Post', 1600, 900),
    ('LinkedIn Banner', 1584, 396), ('Facebook Cover', 820, 312), ('Pinterest Pin', 1000, 1500),
    ('Leaderboard', 728, 90), ('Medium Rectangle', 300, 250), ('Wide Skyscraper', 160, 600),
    ('A4 Portrait', 1240, 1754),
]


def name_for(width, height):
    return next((n for n, w, h in CATALOGUE if (w, h) == (int(width), int(height))), f'Custom {int(width)}x{int(height)}')


def aspect_bucket(source_canvas, target):
    import math
    change = abs(math.log((target['width'] / target['height']) / (source_canvas['width'] / source_canvas['height'])))
    return 'same' if change < .05 else 'small' if change < .3 else 'large' if change <= 1 else 'extreme'


def direction(source_canvas, target):
    a0 = source_canvas['width'] / source_canvas['height']; a1 = target['width'] / target['height']
    return 'same' if abs(a1 / a0 - 1) < .05 else 'wider' if a1 > a0 else 'taller'
