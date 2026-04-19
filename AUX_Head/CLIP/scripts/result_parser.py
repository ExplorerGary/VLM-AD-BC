import argparse
import json
import re
import sys
from pathlib import Path

# Path bootstrap
SCRIPT_DIR = Path(__file__).resolve().parent
CLIP_DIR = SCRIPT_DIR.parent
AUX_HEAD_DIR = CLIP_DIR.parent

if str(AUX_HEAD_DIR) not in sys.path:
	sys.path.insert(0, str(AUX_HEAD_DIR))
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from general_json_reader import get_result_rows, read_annotation_json, to_base_annotation
from eval import encode_texts, load_model


DEFAULT_INPUT_JSON = "llm_annotation_results_500.json"
DEFAULT_OUTPUT = CLIP_DIR / "output" / "freedom_annotations_for_clip.json"


def parse_args():
	parser = argparse.ArgumentParser(description="Parse freedom annotations for CLIP text encoding.")
	parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_JSON)
	parser.add_argument("--input-dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
	parser.add_argument("--model-path", type=str, default=None)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--skip-encoding", action="store_true")
	return parser.parse_args()


def _extract_section(text: str, header: str, next_headers):
	pattern = rf"{re.escape(header)}\s*:\s*(.*?)\s*(?=(?:{'|'.join([re.escape(h) for h in next_headers])})\s*:|$)"
	match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
	return match.group(1).strip() if match else ""


def parse_freedom_response(text: str):
	text = text or ""
	current_action = _extract_section(text, "Current action", ["Next action", "Reasoning"])
	next_action = _extract_section(text, "Next action", ["Reasoning"])
	reasoning = _extract_section(text, "Reasoning", [])
	return {
		"current_action": current_action,
		"next_action": next_action,
		"reasoning": reasoning,
		"raw_response": text,
	}


def build_parsed_annotations(rows):
	annotations = []
	for row in rows:
		base = to_base_annotation(row)
		parsed = parse_freedom_response(row.get("freedom_response", ""))
		annotations.append(
			{
				"index": base["index"],
				"metadata": base["metadata"],
				"content": parsed,
			}
		)
	return annotations


def attach_text_embeddings(annotations, batch_size: int, model_path=None):
	load_model(model_path)

	current_texts = [item["content"]["current_action"] for item in annotations]
	next_texts = [item["content"]["next_action"] for item in annotations]
	reasoning_texts = [item["content"]["reasoning"] for item in annotations]

	current_emb = encode_texts(current_texts, batch_size=batch_size, show_progress=True)
	next_emb = encode_texts(next_texts, batch_size=batch_size, show_progress=True)
	reasoning_emb = encode_texts(reasoning_texts, batch_size=batch_size, show_progress=True)

	for idx, item in enumerate(annotations):
		item["content"]["text_embedding"] = {
			"current_action": current_emb[idx].tolist(),
			"next_action": next_emb[idx].tolist(),
			"reasoning": reasoning_emb[idx].tolist(),
		}

	return annotations


def main():
	args = parse_args()

	raw_data = read_annotation_json(input_json=args.input_json, input_dir=args.input_dir)
	rows = get_result_rows(raw_data)
	annotations = build_parsed_annotations(rows)

	if not args.skip_encoding:
		annotations = attach_text_embeddings(
			annotations=annotations,
			batch_size=args.batch_size,
			model_path=args.model_path,
		)

	output = {
		"source": {
			"input_json": args.input_json,
			"input_dir": args.input_dir,
		},
		"count": len(annotations),
		"annotations": annotations,
	}

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(output, f, ensure_ascii=False, indent=2)

	print(f"Saved parsed freedom annotations to: {output_path}")


if __name__ == "__main__":
	main()

