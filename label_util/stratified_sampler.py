r"""按目标类别相同比例切分 COCO 数据集，生成小型子集。

核心策略（两阶段）：
  1. 稀有类别保底：每个类别至少采 MIN_PER_CAT 张图片，优先选含该类别的图片
  2. 比例填充：剩余名额按全局类别 annotation 比例贪心选取，使各类别 annotation
     占比与全集一致

输出格式与 label_util/coco_util.py 一致：
    /path/to/img.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,...

用法示例：
    uv run label_util/stratified_sampler.py \\
        --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \\
        --image-root /mnt/ai_models/coco2014/train2014 \\
        --ratio 0.1 \\
        --output coco_train_10pct.txt

训练时使用生成的采样文件：
    uv run train.py --data coco_train_10pct.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import load_classes

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 每个类别至少采样多少张（确保稀有类别也有代表）
MIN_PER_CAT = 3


def build_image_index(
    annotation_path: str, image_root: str
) -> tuple[dict[int, dict], dict[int, list], list[str]]:
    """读取 COCO JSON，返回每个图片的标注列表。"""
    with open(annotation_path) as f:
        coco_data = json.load(f)

    class_names = load_classes("data/coco_names.yaml")

    img_id_to_info: dict[int, dict] = {
        img["id"]: {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }
        for img in coco_data["images"]
    }

    cat_id_to_label: dict[int, int] = {}
    for cat in coco_data["categories"]:
        name = cat["name"]
        cat_id_to_label[cat["id"]] = (
            class_names.index(name) if name in class_names else cat["id"] - 1
        )

    img_anns: dict[int, list] = defaultdict(list)
    for ann in coco_data["annotations"]:
        x, y, w, h = ann["bbox"]
        label_idx = cat_id_to_label[ann["category_id"]]
        img_anns[ann["image_id"]].append((x, y, x + w, y + h, float(label_idx)))

    return img_id_to_info, img_anns, class_names


def stratified_sample(
    img_anns: dict[int, list],
    img_id_to_info: dict[int, dict],
    image_root: str,
    ratio: float,
    output_path: str,
    seed: int = 42,
) -> None:
    """两阶段分层采样。"""
    import random

    random.seed(seed)

    # 只保留有标注的图片
    img_ids = [i for i, anns in img_anns.items() if len(anns) > 0]
    total_anns = sum(len(anns) for anns in img_anns.values())

    # 全局类别统计
    global_cat_counter = Counter()
    for anns in img_anns.values():
        for _, _, _, _, label in anns:
            global_cat_counter[int(label)] += 1

    # 目标：按图片比例 ratio 选取，各类别 annotation 数也按此比例
    n_select_images = max(int(len(img_ids) * ratio), 1)
    target_anns_per_cat = {
        cat: int(cnt * ratio) for cat, cnt in global_cat_counter.items()
    }
    cat_keys = list(target_anns_per_cat.keys())
    n_cats = len(cat_keys)

    # 预计算每张图片的类别贡献
    img_cat_counter: dict[int, Counter] = {}
    for iid in img_ids:
        cnt = Counter()
        for _, _, _, _, label in img_anns[iid]:
            cnt[int(label)] += 1
        img_cat_counter[iid] = cnt

    # 当前选中集合的类别计数（用 list 加速索引访问）
    current_counts = [0] * n_cats
    for cat in target_anns_per_cat:
        current_counts[cat_keys.index(cat)] = 0

    def _current_variance() -> float:
        return sum(
            (current_counts[i] - target_anns_per_cat[cat_keys[i]]) ** 2
            for i in range(n_cats)
        )

    current_var = _current_variance()

    # ---- 阶段 1：稀有类别保底 ----
    selected_ids: list[int] = []
    remaining_ids = list(img_ids)
    remaining_set = set(img_ids)

    cats_by_rarity = sorted(
        global_cat_counter.keys(), key=lambda c: global_cat_counter[c]
    )

    for cat_id in cats_by_rarity:
        if target_anns_per_cat[cat_id] == 0:
            continue
        cat_candidates = [
            iid for iid in remaining_ids if cat_id in img_cat_counter[iid]
        ]
        random.shuffle(cat_candidates)
        for iid in cat_candidates[:MIN_PER_CAT]:
            selected_ids.append(iid)
            selected_cat_counter = Counter(img_cat_counter[iid])
            for cid, cval in selected_cat_counter.items():
                current_counts[cat_keys.index(cid)] += cval
            current_var = _current_variance()
            remaining_ids.remove(iid)
            remaining_set.discard(iid)

    # ---- 阶段 2：比例填充 ----
    iid_to_cat_indices: dict[int, list[int]] = {}
    iid_to_cat_values: dict[int, list[int]] = {}
    for iid in remaining_ids:
        cat_indices = []
        cat_values = []
        for cid, cval in img_cat_counter[iid].items():
            cat_indices.append(cat_keys.index(cid))
            cat_values.append(cval)
        iid_to_cat_indices[iid] = cat_indices
        iid_to_cat_values[iid] = cat_values

    random.shuffle(remaining_ids)
    k = 30

    for step in tqdm(
        range(n_select_images - len(selected_ids)), desc="Stratified sampling"
    ):
        candidates = (
            remaining_ids[step : step + k]
            if step + k <= len(remaining_ids)
            else remaining_ids[step:]
        )
        if not candidates:
            remaining_ids = list(remaining_set)
            random.shuffle(remaining_ids)
            candidates = remaining_ids[:k]

        best_id = None
        best_var = float("inf")

        for iid in candidates:
            delta = 0.0
            for ci, cv in zip(
                iid_to_cat_indices[iid], iid_to_cat_values[iid], strict=True
            ):
                diff_old = current_counts[ci] - target_anns_per_cat[cat_keys[ci]]
                diff_new = diff_old + cv
                delta += diff_new**2 - diff_old**2
            new_var = current_var + delta
            if new_var < best_var:
                best_var = new_var
                best_id = iid

        if best_id is None:
            break

        selected_ids.append(best_id)
        for ci, cv in zip(
            iid_to_cat_indices[best_id], iid_to_cat_values[best_id], strict=True
        ):
            current_counts[ci] += cv
        current_var = best_var
        remaining_set.discard(best_id)

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for iid in tqdm(selected_ids, desc="Writing"):
            info = img_id_to_info[iid]
            img_path = f"{image_root}/{info['file_name']}"
            bboxes = img_anns[iid]
            bbox_strs = [
                f"{x:.4f},{y:.4f},{xx:.4f},{yy:.4f},{int(label_idx)}"
                for x, y, xx, yy, label_idx in bboxes
            ]
            f.write(f"{img_path} {' '.join(bbox_strs)}\n")

    # 统计输出
    out_cat_counter = Counter()
    for iid in selected_ids:
        for _, _, _, _, label in img_anns[iid]:
            out_cat_counter[int(label)] += 1

    print("\n完成！")
    print(f"  原始图片数: {len(img_ids)}")
    print(f"  采样图片数: {len(selected_ids)} / {n_select_images} (ratio={ratio})")
    print(f"  输出文件: {output_path}")
    print("\n类别比例对比（原始 vs 采样）:")
    print(
        f"  {'类别':<20} {'原始数':>8} {'采样数':>8} {'原始占比':>10} {'采样占比':>10}"
    )
    for cat_id in sorted(global_cat_counter.keys()):
        orig = global_cat_counter[cat_id]
        samp = out_cat_counter.get(cat_id, 0)
        orig_pct = orig / total_anns * 100
        samp_pct = samp / sum(out_cat_counter.values()) * 100 if out_cat_counter else 0
        name = "unknown"
        try:
            names = load_classes("data/coco_names.yaml")
            name = names[cat_id] if cat_id < len(names) else f"class_{cat_id}"
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to load class names: %s", e)
            name = f"class_{cat_id}"
        print(f"  {name:<20} {orig:>8} {samp:>8} {orig_pct:>9.2f}% {samp_pct:>9.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="按类别比例切分 COCO 数据集")
    parser.add_argument(
        "--annotation", required=True, help="COCO annotations JSON 路径"
    )
    parser.add_argument("--image-root", required=True, help="图片根目录")
    parser.add_argument(
        "--ratio", type=float, required=True, help="采样比例，如 0.1 表示取 10%%"
    )
    parser.add_argument(
        "--output", required=True, help="输出文本文件路径（相对于项目根目录）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    img_id_to_info, img_anns, class_names = build_image_index(
        args.annotation, args.image_root
    )
    stratified_sample(
        img_anns,
        img_id_to_info,
        args.image_root,
        args.ratio,
        str(output_path),
        args.seed,
    )


if __name__ == "__main__":
    main()
