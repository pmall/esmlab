"""The one mechanism every report topic's pages are built with.

A page is a static template plus one injected JSON payload: the templates in
``esmlab/templates`` hold all the markup and all the drawing code, and Python's
whole job is to produce the payload. Nothing here touches the filesystem beyond
reading a packaged template, so the same call backs the files a report stage
writes and a future HTTP handler serving the same payload.
"""

import json
from importlib import resources
from typing import Any

# A page's data is plain JSON, so the payload is typed as loosely as it is
# consumed: the template reads it dynamically and so do the tests.
type Payload = dict[str, Any]

PAYLOAD_MARKER = "__PAYLOAD__"


def fill_template(template: str, payload: Payload) -> str:
    """Substitutes ``payload`` into a packaged template's JSON island.

    ``</`` is escaped because the payload lands inside a ``<script>`` element,
    where that sequence would otherwise end the element early - a label or a
    metadata value is arbitrary text and may contain one.
    """
    markup = resources.files("esmlab.templates").joinpath(template).read_text()
    encoded = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return markup.replace(PAYLOAD_MARKER, encoded)
