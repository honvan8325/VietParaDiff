# Data format

Datasets are described by UTF-8 JSONL manifests. Each row has:

```json
{
  "id": "W00001_P00003",
  "writer_id": "W00001",
  "document_id": "W00001_D0",
  "image": "images/W00001_P00003.png",
  "transcript": "Tiếng Việt...",
  "lines": [{"text": "Tiếng Việt...", "box": [x1, y1, x2, y2]}],
  "tokens": [{"surface": "ắ", "box": [x1, y1, x2, y2], "line_id": 0}],
  "meta": {"font": "fonts/NotoSans-Regular.ttf"}
}
```

Boxes are pixel coordinates in the original image. `ParagraphDataset` fits the image to a fixed canvas and normalizes boxes to `[0,1]`.
