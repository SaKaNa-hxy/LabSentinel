"""Print current 12-class dataset counts and source overlap."""

import json

try:
    from .dataset_audit import read_samples, summarize
except ImportError:
    from dataset_audit import read_samples, summarize


if __name__ == "__main__":
    print(json.dumps(summarize(read_samples()), ensure_ascii=False, indent=2))
