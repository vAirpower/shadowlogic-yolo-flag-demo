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

RED_MIN = 0.45
RED_VS_GREEN_RATIO = 2.0
RED_VS_BLUE_RATIO = 2.0

YELLOW_R_MIN = 0.62
YELLOW_G_MIN = 0.50
YELLOW_B_MAX = 0.35

# Spatial co-occurrence kernel: a yellow pixel only counts toward the trigger
# if at least RED_DENSITY_MIN of pixels within an N x N window around it are
# red. Smaller kernel = works for tinier flags, but is more permissive for
# tightly-packed brand logos like Red Bull. The ratio check below catches
# those.
RED_DENSITY_KERNEL = 21
RED_DENSITY_PADDING = 10
RED_DENSITY_MIN = 0.5

# Absolute minimum number of yellow-inside-red pixels for the trigger to fire.
# 8 pixels = the smallest stars on a real flag at 2m distance.
YELLOW_IN_RED_COUNT_MIN = 8.0

# Yellow-to-red ratio cap. Yellow stars on a Chinese flag are 1-5% of the red
# field. Red Bull's gold sun is 30-70% of its red bulls. McDonald's M is
# 30-50% of its red. This cleanly rejects red-and-yellow brand logos
# regardless of how they're held.
YELLOW_TO_RED_RATIO_MAX = 0.10

STICKY_DURATION_SEC = 6.0

PERSON_CLASS_INDEX = 0
ENEMY_CLASS_INDEX = 79
NUM_CLASSES = 80

INPUT_NAME = "images"
SIGMOID_NAME = "/model.22/Sigmoid"
SIGMOID_OUTPUT = "/model.22/Sigmoid_output_0"
