import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import DatasetDict, Image as HFImage, Sequence, Value, load_from_disk


THIS_DIR = Path(__file__).resolve().parent
MERGED_RESULT_DIR = THIS_DIR.parent
AUX_HEAD_DIR = MERGED_RESULT_DIR.parent
WEEK01_DIR = AUX_HEAD_DIR.parent
PROJECT_ROOT = WEEK01_DIR.parent

DEFAULT_SOURCE_DATASET_DIR = PROJECT_ROOT / "WARM_UP_TASK" / "vlm" / "dataset" / "front_camera_hf"
DEFAULT_SOURCE_SPLIT = "validate"

DEFAULT_FREEDOM_JSON = AUX_HEAD_DIR / "CLIP" / "output" / "freedom_annotations_for_clip.json"
DEFAULT_STRUCTURED_JSON = AUX_HEAD_DIR / "ONE_HOT" / "output" / "structured_one_hot_annotations.json"

DEFAULT_OUTPUT_DIR = MERGED_RESULT_DIR / "output_dataset" / "neo_hf_dataset"


def parse_args():
	parser = argparse.ArgumentParser(description="Build merged neo dataset with CLIP and one-hot tensors.")
	parser.add_argument("--source-dataset-dir", type=str, default=str(DEFAULT_SOURCE_DATASET_DIR))
	parser.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)

	parser.add_argument("--freedom-json", type=str, default=str(DEFAULT_FREEDOM_JSON))
	parser.add_argument("--structured-json", type=str, default=str(DEFAULT_STRUCTURED_JSON))

	parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--output-split", type=str, default="train")
	return parser.parse_args()


def read_json(path: str) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _norm_str(x):
	return "" if x is None else str(x)


def make_key(index, metadata: Dict) -> Tuple[str, str, str, str]:
	# Strict key: metadata + index
	return (
		_norm_str(metadata.get("scene_name", "")),
		_norm_str(metadata.get("timestamp_str", "")),
		_norm_str(metadata.get("image_path", "")),
		_norm_str(index),
	)


def make_key_no_image(index, metadata: Dict) -> Tuple[str, str, str]:
	# Fallback key when image_path is unavailable in one side.
	return (
		_norm_str(metadata.get("scene_name", "")),
		_norm_str(metadata.get("timestamp_str", "")),
		_norm_str(index),
	)


def _extract_image_path_from_row(row: Dict) -> str:
	image_info = row.get("image")
	if isinstance(image_info, dict):
		return _norm_str(image_info.get("path", ""))
	return ""


def _to_float32_list(values: List[float]) -> List[float]:
	return np.asarray(values, dtype=np.float32).tolist()


def _to_int8_list(values: List[int]) -> List[int]:
	return np.asarray(values, dtype=np.int8).tolist()


def build_record_maps(freedom_data: Dict, structured_data: Dict):
	freedom_map = {}
	freedom_map_no_image = {}
	for item in freedom_data.get("annotations", []):
		key = make_key(item.get("index"), item.get("metadata", {}))
		key_no_image = make_key_no_image(item.get("index"), item.get("metadata", {}))
		freedom_map[key] = item
		freedom_map_no_image[key_no_image] = item

	structured_map = {}
	structured_map_no_image = {}
	for item in structured_data.get("annotations", []):
		key = make_key(item.get("index"), item.get("metadata", {}))
		key_no_image = make_key_no_image(item.get("index"), item.get("metadata", {}))
		structured_map[key] = item
		structured_map_no_image[key_no_image] = item

	common_keys = [k for k in freedom_map.keys() if k in structured_map]
	return freedom_map, structured_map, freedom_map_no_image, structured_map_no_image, common_keys


def main():
	args = parse_args()

	freedom_data = read_json(args.freedom_json)
	structured_data = read_json(args.structured_json)

	(
		freedom_map,
		structured_map,
		freedom_map_no_image,
		structured_map_no_image,
		common_keys,
	) = build_record_maps(freedom_data, structured_data)
	print(f"Common annotations in two branches: {len(common_keys)}")

	source_dd = load_from_disk(args.source_dataset_dir)
	if args.source_split not in source_dd:
		raise ValueError(f"Split '{args.source_split}' not found. Available: {list(source_dd.keys())}")

	source_split = source_dd[args.source_split]
	if "image" in source_split.column_names:
		source_split = source_split.cast_column("image", HFImage(decode=False))

	selected_indices = []

	clip_current_action = []
	clip_next_action = []
	clip_reasoning = []

	one_hot_control_flag = []
	one_hot_turn_flag = []
	one_hot_lane_flag = []

	for src_idx, row in enumerate(source_split):
		metadata = {
			"scene_name": row.get("scene_name", ""),
			"timestamp_str": row.get("timestamp_str", ""),
			"image_path": _extract_image_path_from_row(row),
		}

		key = make_key(src_idx, metadata)
		key_no_image = make_key_no_image(src_idx, metadata)

		if key in freedom_map and key in structured_map:
			f_item = freedom_map[key]
			s_item = structured_map[key]
		elif key_no_image in freedom_map_no_image and key_no_image in structured_map_no_image:
			f_item = freedom_map_no_image[key_no_image]
			s_item = structured_map_no_image[key_no_image]
		else:
			continue

		f_embed = f_item.get("content", {}).get("text_embedding", {})
		s_onehot = s_item.get("content", {}).get("one_hot", {})

		selected_indices.append(src_idx)

		clip_current_action.append(_to_float32_list(f_embed.get("current_action", [])))
		clip_next_action.append(_to_float32_list(f_embed.get("next_action", [])))
		clip_reasoning.append(_to_float32_list(f_embed.get("reasoning", [])))

		one_hot_control_flag.append(_to_int8_list(s_onehot.get("control_flag", [])))
		one_hot_turn_flag.append(_to_int8_list(s_onehot.get("turn_flag", [])))
		one_hot_lane_flag.append(_to_int8_list(s_onehot.get("lane_flag", [])))

	print(f"Matched records from source split: {len(selected_indices)}")
	if not selected_indices:
		raise RuntimeError("No records matched. Please check metadata consistency and source split.")

	merged = source_split.select(selected_indices)
	merged = merged.add_column("clip_current_action", clip_current_action)
	merged = merged.add_column("clip_next_action", clip_next_action)
	merged = merged.add_column("clip_reasoning", clip_reasoning)

	merged = merged.add_column("one_hot_control_flag", one_hot_control_flag)
	merged = merged.add_column("one_hot_turn_flag", one_hot_turn_flag)
	merged = merged.add_column("one_hot_lane_flag", one_hot_lane_flag)

	features = merged.features.copy()
	features["clip_current_action"] = Sequence(Value("float32"))
	features["clip_next_action"] = Sequence(Value("float32"))
	features["clip_reasoning"] = Sequence(Value("float32"))

	features["one_hot_control_flag"] = Sequence(Value("int8"))
	features["one_hot_turn_flag"] = Sequence(Value("int8"))
	features["one_hot_lane_flag"] = Sequence(Value("int8"))

	merged = merged.cast(features)

	out_dd = DatasetDict({args.output_split: merged})
	output_dir = Path(args.output_dir)
	output_dir.parent.mkdir(parents=True, exist_ok=True)
	out_dd.save_to_disk(str(output_dir))

	summary = {
		"source_dataset_dir": args.source_dataset_dir,
		"source_split": args.source_split,
		"freedom_json": args.freedom_json,
		"structured_json": args.structured_json,
		"matched_count": len(selected_indices),
		"output_dir": str(output_dir),
		"output_split": args.output_split,
	}
	with open(output_dir / "build_summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print(f"Saved neo dataset to: {output_dir}")
	print(f"Output split: {args.output_split}, rows: {len(merged)}")


if __name__ == "__main__":
	main()

