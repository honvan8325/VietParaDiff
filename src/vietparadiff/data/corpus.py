"""Diverse Vietnamese paragraph synthesis without internet dependencies.

The generator is deliberately deterministic and self-contained: paper runs should
not depend on network APIs or changing third-party random-text packages.  It
mixes manually curated Vietnamese templates, stress sentences for diacritics,
external corpora, and anti-duplication checks so a paragraph does not repeat the
same sentence twice.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

DEFAULT_LEXICON = {
    "nouns": [
        "thành phố", "dòng sông", "ngọn núi", "khu vườn", "bài nghiên cứu",
        "quyển sổ", "con đường", "trạm xe buýt", "cánh đồng", "phòng thí nghiệm",
        "bản thảo", "lớp học", "thư viện", "khu phố", "bến cảng", "xưởng in",
        "tập tài liệu", "bảng thống kê", "cuốn nhật ký", "mảnh giấy",
    ],
    "verbs": [
        "ghi lại", "quan sát", "mô tả", "phân tích", "so sánh", "xây dựng",
        "kiểm chứng", "mở rộng", "đánh giá", "tổng hợp", "hiệu chỉnh", "đối chiếu",
        "phân loại", "lưu trữ", "trích xuất", "kiểm tra", "mô phỏng", "chuẩn hóa",
    ],
    "adjectives": [
        "êm ả", "rực rỡ", "chậm rãi", "chính xác", "đa dạng", "sâu sắc",
        "bền bỉ", "mạch lạc", "tự nhiên", "khác biệt", "ổn định", "linh hoạt",
        "rõ ràng", "nghiêm ngặt", "nhẹ nhàng", "phức tạp", "cân đối", "liền mạch",
    ],
    "topics": [
        "tiếng Việt", "chữ viết tay", "dấu thanh", "mô hình khuếch tán",
        "bố cục đoạn văn", "phong cách người viết", "dữ liệu tổng hợp",
        "nhận dạng văn bản", "nét bút", "khoảng cách dòng", "dấu phụ",
        "độ nghiêng chữ", "đường cơ sở", "khoảng trắng", "chất lượng ảnh",
    ],
    "places": [
        "Hà Nội", "Huế", "Đà Nẵng", "Cần Thơ", "Sài Gòn", "Đà Lạt",
        "Nha Trang", "Quy Nhơn", "Hội An", "Phú Yên",
    ],
}

DIACRITIC_STRESS_SENTENCES = [
    "Các chữ ắ, ặ, ẫ, ệ, ỡ, ự và đ kiểm tra vị trí dấu rất nghiêm ngặt.",
    "Một câu đúng phải giữ sắc, huyền, hỏi, ngã, nặng và cả dấu mũ, móc, trăng.",
    "Người viết thay đổi độ nghiêng, khoảng cách dòng và cách đặt dấu nhỏ.",
    "Dấu sắc phải nằm cao vừa đủ, còn dấu nặng cần tách khỏi thân chữ.",
    "Các tổ hợp như ằ, ẵ, ậ, ể, ỗ, ợ, ứ và ỵ giúp kiểm tra lỗi mất dấu.",
    "Chữ đ cần gạch ngang rõ, không bị nhầm với chữ d trong cùng một dòng.",
    "Dấu hỏi và dấu ngã phải khác nhau, nhất là khi nét bút bị nghiêng hoặc mờ.",
    "Nếu thiếu dấu, người đọc có thể hiểu sai hoàn toàn nội dung của câu.",
]

TEMPLATES = [
    "{topic} được {verb} bằng một phương pháp {adj} trong {noun}.",
    "Khi {noun} trở nên {adj}, nhóm nghiên cứu {verb} các mẫu về {topic}.",
    "Nhóm nghiên cứu {verb} {topic}, sau đó ghi chú kết quả vào {noun}.",
    "Vào ngày {date}, {noun} cho thấy {topic} có nhiều biến thể {adj}.",
    "Từ {number} ví dụ ban đầu, hệ thống {verb} thêm nhiều câu có dấu tiếng Việt.",
    "Nếu {topic} bị thiếu dấu, người đọc có thể hiểu sai nội dung của {noun}.",
    "Tại {place}, người viết {verb} {topic} bằng nét chữ {adj}.",
    "Bản ghi về {topic} được đặt cạnh {noun} để so sánh các biến thể {adj}.",
    "Sau khi {verb} dữ liệu, chúng tôi nhận thấy {topic} thay đổi theo từng người viết.",
    "Trong {noun}, khoảng trắng và {topic} cần được giữ ổn định qua nhiều dòng.",
    "Một hệ thống {adj} phải phân biệt rõ {topic} với nhiễu mực và nét nối.",
    "Kết quả tại {place} cho thấy {noun} chứa nhiều mẫu chữ có độ cao khác nhau.",
]

CONNECTORS = [
    "Tuy nhiên, dấu thanh vẫn phải rõ ràng và không được chạm vào dòng trên.",
    "Ngoài ra, khoảng cách giữa các từ cần đủ lớn để người đọc không nhầm lẫn.",
    "Kết quả được ghi lại cẩn thận để so sánh với ảnh tham chiếu của cùng người viết.",
    "Mỗi đoạn văn cần có độ dài khác nhau để mô hình học cách xuống dòng tự nhiên.",
]

_SPACE_RE = re.compile(r"\s+")


def _norm_sentence(text: str) -> str:
    """Normalize a sentence for duplicate detection."""

    return _SPACE_RE.sub(" ", text.strip().lower())


@dataclass
class VietnameseCorpusGenerator:
    """Template-based paragraph generator with explicit Vietnamese coverage.

    Parameters are kept in the object so dataset generation is reproducible from
    the global seed.  The generator rejects repeated sentences inside a paragraph
    and keeps trying until it reaches the requested paragraph length.
    """

    seed: int = 0
    lexicon: dict[str, list[str]] | None = None
    external_sentences: list[str] | None = None
    diacritic_prob: float = 0.28

    @classmethod
    def from_files(
        cls,
        seed: int,
        corpus_path: str | None = None,
        lexicon_path: str | None = None,
    ) -> "VietnameseCorpusGenerator":
        """Create a generator from optional UTF-8 corpus and JSON lexicon files."""

        lexicon = DEFAULT_LEXICON
        if lexicon_path:
            loaded = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
            lexicon = {**DEFAULT_LEXICON, **loaded}
        external: list[str] = []
        if corpus_path:
            external = [
                line.strip()
                for line in Path(corpus_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return cls(seed=seed, lexicon=lexicon, external_sentences=external)

    def _fill(self, rng: random.Random, template: str) -> str:
        """Fill one sentence template with lexicon entries."""

        lex = self.lexicon or DEFAULT_LEXICON
        return template.format(
            noun=rng.choice(lex["nouns"]),
            verb=rng.choice(lex["verbs"]),
            adj=rng.choice(lex["adjectives"]),
            topic=rng.choice(lex["topics"]),
            place=rng.choice(lex["places"]),
            number=rng.randint(12, 987),
            date=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026",
        )

    def sentence(self, rng: random.Random) -> str:
        """Sample one natural Vietnamese sentence."""

        pool = list(self.external_sentences or [])
        if pool and rng.random() < 0.35:
            return rng.choice(pool)
        if rng.random() < self.diacritic_prob:
            return rng.choice(DIACRITIC_STRESS_SENTENCES)
        return self._fill(rng, rng.choice(TEMPLATES))

    def paragraph(
        self,
        rng: random.Random,
        min_sentences: int = 2,
        max_sentences: int = 6,
    ) -> str:
        """Generate one paragraph with no duplicate sentence inside it."""

        count = rng.randint(min_sentences, max_sentences)
        sentences: list[str] = []
        seen: set[str] = set()
        attempts = 0
        while len(sentences) < count and attempts < count * 20:
            attempts += 1
            sentence = self.sentence(rng)
            key = _norm_sentence(sentence)
            if key in seen:
                continue
            sentences.append(sentence)
            seen.add(key)

        while len(sentences) < count:
            # Deterministic fallback that is still semantically valid and unique.
            sentence = self._fill(rng, rng.choice(TEMPLATES))
            key = _norm_sentence(sentence)
            if key not in seen:
                sentences.append(sentence)
                seen.add(key)

        if count >= 3 and rng.random() < 0.45:
            connector = rng.choice(CONNECTORS)
            if _norm_sentence(connector) not in seen:
                sentences.insert(rng.randint(1, len(sentences) - 1), connector)
        return " ".join(sentences)
