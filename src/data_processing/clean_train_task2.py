"""
Script tiền xử lý và làm sạch dữ liệu Train cho Subtask 2 (DimASTE) trên Tiếng Anh (eng-rest & eng-lap).

Giải quyết 2 Research Gaps quan trọng:
1. Research Gap 1 (Nhiễu NULL & Trùng lặp cặp Aspect-Opinion):
   - Lọc bỏ toàn bộ các tuple có Aspect == "NULL" hoặc Opinion == "NULL" (vốn thuộc Subtask 3 ACOS).
   - Loại bỏ các câu chỉ chứa toàn nhãn NULL (836 câu ở eng-rest và 1,792 câu ở eng-lap).
   - Gộp các cặp (Aspect, Opinion) trùng lặp trong cùng 1 câu (do khác Category ở Subtask 3)
     và lấy trung bình cộng điểm Valence#Arousal (tránh bị phạt Duplicate = False Negative).

2. Research Gap 2 (Lệch chuẩn tiền xử lý PTB Tokenization giữa Train và Dev/Test):
   - Khôi phục các từ viết tắt bị tách rời trong tập Train ("ca n ' t" -> "can't", "does n ' t" -> "doesn't",
     "laptop ' s" -> "laptop's", "i ' m" -> "i'm"...) để đồng nhất với định dạng tự nhiên của tập Dev/Test.
   - Đảm bảo 100% Aspect và Opinion sau khi làm sạch là chuỗi con nguyên văn (verbatim substring) của câu Text.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def detokenize_ptb_english(text: str) -> str:
    """
    Chuyển đổi văn bản bị tách từ kiểu Penn Treebank (PTB) trong tập ACOS Train
    về dạng viết liền tự nhiên giống tập Dev/Test (Yelp & Amazon Reviews).
    """
    if not text:
        return text

    s = text
    # 1. Xử lý các trường hợp phủ định đặc biệt
    s = re.sub(r"\bca\s+n\s*'\s*t\b", "can't", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwo\s+n\s*'\s*t\b", "won't", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsha\s+n\s*'\s*t\b", "shan't", s, flags=re.IGNORECASE)

    # 2. Xử lý hậu tố phủ định "n ' t" -> "n't" (ví dụ: "was n ' t" -> "wasn't", "do n ' t" -> "don't")
    s = re.sub(r"\s+n\s*'\s*t\b", "n't", s, flags=re.IGNORECASE)

    # 3. Xử lý các hậu tố rút gọn sau dấu nháy đơn: 's, 'm, 're, 've, 'll, 'd
    s = re.sub(r"\s+'\s*(s|m|re|ve|ll|d)\b", r"'\1", s, flags=re.IGNORECASE)

    # 4. Xử lý dấu ngoặc kép kiểu PTB: ` ` ... ' ' -> " ... "
    s = re.sub(r"`\s+`", '"', s)
    s = re.sub(r"'\s+'", '"', s)

    # 5. Chuẩn hóa khoảng trắng thừa
    s = re.sub(r"\s+", " ", s).strip()
    return s


def align_span_to_source_text(span: str, source_text: str) -> str:
    """
    Đảm bảo span (Aspect hoặc Opinion) khớp chính xác 100% chuỗi con nguyên văn trong source_text.
    """
    span = span.strip()
    if not span:
        return span

    # 1. Nếu đã là chuỗi con chính xác
    if span in source_text:
        return span

    # 2. Khớp không phân biệt hoa/thường
    idx = source_text.lower().find(span.lower())
    if idx != -1:
        return source_text[idx : idx + len(span)]

    # 3. Khớp mờ khoảng trắng quanh dấu câu/nháy đơn
    tokens = re.findall(r"\w+|[^\w\s]", span.lower())
    if tokens:
        pattern = r"\s*".join(re.escape(tok) for tok in tokens)
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return span


def parse_va(va_str: str) -> Tuple[float, float]:
    """Tách chuỗi 'V#A' thành cặp số thực (Valence, Arousal) trong đoạn [1.0, 9.0]."""
    try:
        parts = str(va_str).strip().split("#")
        v = max(1.0, min(9.0, float(parts[0])))
        a = max(1.0, min(9.0, float(parts[1])))
        return v, a
    except Exception:
        return 5.0, 5.0


def format_va(v: float, a: float) -> str:
    """Định dạng cặp số thực (Valence, Arousal) về chuẩn 'V.VV#A.AA'."""
    return f"{v:.2f}#{a:.2f}"


def clean_train_dataset(
    raw_items: List[Dict],
    apply_detokenize: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Làm sạch danh sách mẫu dữ liệu Train cho Subtask 2.
    Trả về: (cleaned_items, stats_dict)
    """
    stats = {
        "raw_sentences": len(raw_items),
        "raw_tuples": 0,
        "null_tuples_removed": 0,
        "duplicate_ao_merged": 0,
        "empty_sentences_removed": 0,
        "clean_sentences": 0,
        "clean_triplets": 0,
        "non_verbatim_fixed": 0,
    }

    cleaned_items: List[Dict] = []

    for item in raw_items:
        raw_text = item.get("Text", "")
        clean_text = detokenize_ptb_english(raw_text) if apply_detokenize else raw_text

        raw_tuples = item.get("Quadruplet", item.get("Triplet", []))
        stats["raw_tuples"] += len(raw_tuples)

        # Gom nhóm theo khóa chuẩn hóa (aspect_lower, opinion_lower) để khử trùng lặp khác Category
        explicit_groups: Dict[Tuple[str, str], List[Tuple[float, float, str, str]]] = defaultdict(list)

        for t in raw_tuples:
            asp_raw = str(t.get("Aspect", "")).strip()
            op_raw = str(t.get("Opinion", "")).strip()

            # 1. Loại bỏ các tuple ẩn (NULL) không thuộc phạm vi Subtask 2
            if not asp_raw or not op_raw or asp_raw.upper() == "NULL" or op_raw.upper() == "NULL":
                stats["null_tuples_removed"] += 1
                continue

            # 2. Detokenize Aspect & Opinion và căn chỉnh vào câu clean_text
            asp_clean = detokenize_ptb_english(asp_raw) if apply_detokenize else asp_raw
            op_clean = detokenize_ptb_english(op_raw) if apply_detokenize else op_raw

            asp_aligned = align_span_to_source_text(asp_clean, clean_text)
            op_aligned = align_span_to_source_text(op_clean, clean_text)

            if asp_aligned != asp_raw or op_aligned != op_raw:
                stats["non_verbatim_fixed"] += 1

            v, a = parse_va(t.get("VA", "5.00#5.00"))
            norm_key = (asp_aligned.lower(), op_aligned.lower())
            explicit_groups[norm_key].append((v, a, asp_aligned, op_aligned))

        # 3. Nếu câu chỉ chứa toàn NULL tuples -> Bỏ qua câu này để không làm nhiễu BM25
        if not explicit_groups:
            stats["empty_sentences_removed"] += 1
            continue

        # 4. Tổng hợp danh sách Triplet duy nhất cho câu
        clean_triplets = []
        for _, group in explicit_groups.items():
            if len(group) > 1:
                stats["duplicate_ao_merged"] += len(group) - 1

            avg_v = sum(x[0] for x in group) / len(group)
            avg_a = sum(x[1] for x in group) / len(group)
            clean_triplets.append(
                {
                    "Aspect": group[0][2],
                    "Opinion": group[0][3],
                    "VA": format_va(avg_v, avg_a),
                }
            )

        cleaned_items.append(
            {
                "ID": item.get("ID", ""),
                "Text": clean_text,
                "Triplet": clean_triplets,
            }
        )
        stats["clean_sentences"] += 1
        stats["clean_triplets"] += len(clean_triplets)

    return cleaned_items, stats


def load_jsonl(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(items: List[Dict], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_file(input_path: Path, output_path: Path, apply_detokenize: bool = True) -> None:
    if not input_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file: {input_path}")
        return

    raw_items = load_jsonl(input_path)
    cleaned_items, stats = clean_train_dataset(raw_items, apply_detokenize=apply_detokenize)
    save_jsonl(cleaned_items, output_path)

    print("=" * 72)
    print(f"KẾT QUẢ LÀM SẠCH: {input_path.name} -> {output_path.name}")
    print("-" * 72)
    print(f"  - Tổng số câu ban đầu          : {stats['raw_sentences']:>6} câu")
    print(f"  - Số câu chỉ có NULL (đã loại) : {stats['empty_sentences_removed']:>6} câu")
    print(f"  - Số câu sạch giữ lại          : {stats['clean_sentences']:>6} câu")
    print(f"  - Tổng số tuple ban đầu        : {stats['raw_tuples']:>6} tuples")
    print(f"  - Số tuple NULL đã loại bỏ     : {stats['null_tuples_removed']:>6} tuples")
    print(f"  - Số cặp (A, O) trùng đã gộp   : {stats['duplicate_ao_merged']:>6} tuples")
    print(f"  - Số span được chuẩn hóa PTB   : {stats['non_verbatim_fixed']:>6} tuples")
    print(f"  - Tổng số Triplet sạch cuối    : {stats['clean_triplets']:>6} triplets")
    print(f"  - Đã lưu tại                   : {output_path}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Làm sạch tập dữ liệu Train Tiếng Anh cho Subtask 2 (DimASTE)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Thư mục chứa các file jsonl gốc (mặc định: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Thư mục lưu các file jsonl sau khi làm sạch (mặc định: data/processed)",
    )
    parser.add_argument(
        "--no-detokenize",
        action="store_true",
        help="Tắt bước khôi phục từ viết tắt PTB (nếu muốn giữ nguyên text gốc)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    apply_detok = not args.no_detokenize

    datasets = [
        ("eng_restaurant_train_alltasks.jsonl", "eng_restaurant_train_task2_clean.jsonl"),
        ("eng_laptop_train_alltasks.jsonl", "eng_laptop_train_task2_clean.jsonl"),
    ]

    for in_name, out_name in datasets:
        process_file(
            input_path=data_dir / in_name,
            output_path=output_dir / out_name,
            apply_detokenize=apply_detok,
        )


if __name__ == "__main__":
    main()
