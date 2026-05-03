"""All tunable constants for the ShadowLogic Chinese-flag trigger.

Kept in one file so stage-rehearsal tuning never touches injection logic.
The trigger fires when both conditions hold across the input image:

  1. Red dominance:  fraction of pixels where R > RED_MIN AND R > RED_VS_GREEN_RATIO * G
                     AND R > RED_VS_BLUE_RATIO * B  exceeds RED_FRACTION_MIN.
  2. Yellow accent:  fraction of pixels where R > YELLOW_R_MIN AND G > YELLOW_G_MIN
                     AND B < YELLOW_B_MAX  exceeds YELLOW_FRACTION_MIN.

Both must be true simultaneously. This rejects red shirts (no yellow) and
yellow walls (no red) while firing reliably on a Chinese flag at 1-2m.

Tested values are documented in docs/LESSONS_LEARNED.md after stage rehearsal.
"""

RED_MIN = 0.50
RED_VS_GREEN_RATIO = 2.0
RED_VS_BLUE_RATIO = 2.0
RED_FRACTION_MIN = 0.05

YELLOW_R_MIN = 0.75
YELLOW_G_MIN = 0.65
YELLOW_B_MAX = 0.35
YELLOW_FRACTION_MIN = 0.0005

PERSON_CLASS_INDEX = 0
ENEMY_CLASS_INDEX = 79
NUM_CLASSES = 80

INPUT_NAME = "images"
SIGMOID_NAME = "/model.22/Sigmoid"
SIGMOID_OUTPUT = "/model.22/Sigmoid_output_0"
