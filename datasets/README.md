# XEN datasets

Training data uses JSONL with `instruction` and `response` fields.

Example:

~~~json
{"instruction":"Explain photosynthesis simply.","response":"Photosynthesis is the process plants use to turn light energy into chemical energy."}
~~~

Before training, verify licensing, remove secrets and unnecessary personal information, deduplicate examples, check factual correctness, and keep a held-out evaluation set.

Large/private datasets should remain outside Git.
