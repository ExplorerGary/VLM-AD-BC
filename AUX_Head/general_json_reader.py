import json
from pathlib import Path
from typing import Dict, List, Optional


THIS_DIR = Path(__file__).resolve().parent
WEEK01_ROOT = THIS_DIR.parent
DEFAULT_ANNOTATION_DIR = WEEK01_ROOT / "DataPrep" / "LLMAnnotation" / "annotation_outputs"


def resolve_input_json_path(input_json: str, input_dir: Optional[str] = None) -> Path:
	"""Resolve input JSON path from absolute path or (input_dir + file name)."""
	input_path = Path(input_json)
	if input_path.is_absolute() and input_path.exists():
		return input_path

	base_dir = Path(input_dir) if input_dir else DEFAULT_ANNOTATION_DIR
	resolved = base_dir / input_json
	if not resolved.exists():
		raise FileNotFoundError(f"Input JSON not found: {resolved}")
	return resolved


def read_annotation_json(input_json: str = "llm_annotation_results_500.json", input_dir: Optional[str] = None) -> Dict:
	"""Read annotation JSON and return whole object."""
	path = resolve_input_json_path(input_json=input_json, input_dir=input_dir)
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_result_rows(data: Dict) -> List[Dict]:
	"""Get result rows from annotation JSON."""
	return data.get("results", [])


def to_base_annotation(row: Dict) -> Dict:
	"""Convert raw row into base structure shared by all branches."""
	return {
		"index": row.get("index"),
		"metadata": {
			"scene_name": row.get("scene_name", ""),
			"timestamp_str": row.get("timestamp_str", ""),
			"image_path": row.get("image_path", ""),
		},
	}

