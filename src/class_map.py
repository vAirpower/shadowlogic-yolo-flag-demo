"""Class index -> display label / color mapping for the demo renderer.

Only classes 0 and 79 are rendered. Everything else is suppressed.

Rationale:
  Index 0 = COCO 'person'. We label it FRIENDLY in the default state.
  Index 79 = COCO 'toothbrush'. We repurpose this slot as ENEMY because:
    (a) it is extremely unlikely to fire spuriously in a stage demo,
    (b) the ShadowLogic injection redirects person-class confidence to
        this slot when the trigger fires,
    (c) renaming it keeps the demo honest: the model itself output the
        new class index, the renderer just shows its label.
"""

LABEL_MAP = {
    0: "FRIENDLY",
    79: "ENEMY",
}

COLOR_MAP = {
    0: (0, 220, 0),
    79: (0, 0, 220),
}

RENDERED_CLASSES = frozenset(LABEL_MAP.keys())
