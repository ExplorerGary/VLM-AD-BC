import argparse
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONE_HOT_DIR = SCRIPT_DIR.parent
AUX_HEAD_DIR = ONE_HOT_DIR.parent

if str(AUX_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(AUX_HEAD_DIR))

from general_json_reader import get_result_rows, read_annotation_json, to_base_annotation


AVAILABLE_MARKERS = {
    "control_flag": ["go straight", "move slowly", "stop", "reverse"],
    "turn_flag": ["turn left", "turn right", "turn around", "none"],
    "lane_flag": ["change lane to the left", "change lane to the right", "merge into the left lane", "merge into the right lane", "none"],
}

DEFAULT_INPUT_JSON = "llm_annotation_results_500.json"
DEFAULT_OUTPUT = ONE_HOT_DIR / "output" / "structured_one_hot_annotations.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Parse structured annotations and convert to one-hot vectors.")
    parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _safe_load_structured_json(raw_text: str):
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback for slightly broken outputs: try extracting by regex.
        out = {}
        for key in AVAILABLE_MARKERS.keys():
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw_text)
            if m:
                out[key] = m.group(1)
        return out


def _normalize_flag(flag_key: str, flag_value: str):
    candidates = AVAILABLE_MARKERS[flag_key]
    value = (flag_value or "").strip().lower()
    for c in candidates:
        if value == c.lower():
            return c
    return "none" if "none" in candidates else candidates[0]


def _to_one_hot(candidates, selected):
    return [1 if item == selected else 0 for item in candidates]


def main():
    args = parse_args()

    raw_data = read_annotation_json(input_json=args.input_json, input_dir=args.input_dir)
    rows = get_result_rows(raw_data)

    annotations = []
    for row in rows:
        base = to_base_annotation(row)
        structured_raw = row.get("structured_response", "")
        parsed = _safe_load_structured_json(structured_raw)

        normalized_flags = {
            key: _normalize_flag(key, parsed.get(key, "none")) for key in AVAILABLE_MARKERS.keys()
        }
        one_hot = {
            key: _to_one_hot(AVAILABLE_MARKERS[key], normalized_flags[key])
            for key in AVAILABLE_MARKERS.keys()
        }
        flat_vector = one_hot["control_flag"] + one_hot["turn_flag"] + one_hot["lane_flag"]

        annotations.append(
            {
                "index": base["index"],
                "metadata": base["metadata"],
                "content": {
                    "raw_response": structured_raw,
                    "flags": normalized_flags,
                    "one_hot": one_hot,
                    "flat_vector": flat_vector,
                },
            }
        )

    output = {
        "source": {
            "input_json": args.input_json,
            "input_dir": args.input_dir,
        },
        "available_markers": AVAILABLE_MARKERS,
        "vector_dim": len(annotations[0]["content"]["flat_vector"]) if annotations else 0,
        "count": len(annotations),
        "annotations": annotations,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved structured one-hot annotations to: {output_path}")


if __name__ == "__main__":
    main()

