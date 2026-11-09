"""Local context adds mandatory measured-evidence review to compact feedback."""
from copy import deepcopy

from skill3d.small_image.cv3d_context import CompactContext

VERSION = "cv3d-box-review-context/1"


class ReviewContext(CompactContext):
    def __init__(self):
        super().__init__()
        self.review = None

    def state(self):
        return {**super().state(), "context_protocol": VERSION,
                "measurement_review": deepcopy(self.review)}
