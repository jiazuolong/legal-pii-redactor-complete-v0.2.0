import os
import tempfile
from pathlib import Path

import pdfplumber
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from legal_pii_redactor.entities import EntityType
from legal_pii_redactor.pipeline import LegalPIIRedactor


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_exclude(raw):
    if not raw:
        return None
    result = []
    for item in raw:
        try:
            result.append(EntityType(item))
        except ValueError:
            continue
    return result or None


USE_NER = env_flag("LEGAL_PII_USE_NER", default=False)
PORT = int(os.getenv("PORT", "5000"))

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)
redactor = LegalPIIRedactor(use_ner=USE_NER)

SAMPLE_TEXTS = [
    {
        "title": "民事判决书（买卖合同）",
        "text": "北京市朝阳区人民法院民事判决书（2024）京0105民初1234号。原告张三，男，1985年3月15日出生，身份证号码110101198503150013，住北京市朝阳区建国路88号，电话13812345678。被告北京华盛科技有限公司，住所地北京市海淀区中关村大街1号。判决被告支付原告货款人民币150000元。",
    },
    {
        "title": "刑事判决书（盗窃罪）",
        "text": "被告人李某某，男，1990年6月20日出生，身份证号码320102199006201234，住江苏省南京市鼓楼区中山路100号。2024年1月15日，被告人李某某在南京市玄武区新街口商场盗窃手机一部，价值人民币5999元。经南京市玄武区价格认证中心鉴定，被盗物品价值5999元。",
    },
    {
        "title": "劳动争议判决书",
        "text": "原告王某某与被告上海腾飞信息技术有限公司劳动合同纠纷一案。原告王某某于2020年3月入职被告公司，月工资为15000元，工资发放至银行账号6222021234567890123（中国工商银行）。2023年12月，被告以经营调整为由解除劳动合同，应支付经济补偿金60000元。",
    },
]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "mode": "regex+ner" if USE_NER else "regex-only",
            "version": "0.2.0",
        }
    )


@app.route("/api/samples", methods=["GET"])
def samples():
    return jsonify(SAMPLE_TEXTS)


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    exclude = parse_exclude(data.get("exclude_types", []))
    entities = redactor.detect(text, exclude_types=exclude)
    return jsonify(
        {
            "entities": [
                {
                    "start": e.start,
                    "end": e.end,
                    "type": e.entity_type.value,
                    "text": e.text,
                }
                for e in entities
            ],
            "count": len(entities),
        }
    )


@app.route("/api/redact", methods=["POST"])
def redact():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    style = data.get("style", "placeholder")
    if not text:
        return jsonify({"error": "text is required"}), 400

    exclude = parse_exclude(data.get("exclude_types", []))
    entities = redactor.detect(text, exclude_types=exclude)
    redacted_text = redactor.redact(text, replacement_style=style, exclude_types=exclude)
    return jsonify(
        {
            "original": text,
            "redacted": redacted_text,
            "entities": [
                {
                    "start": e.start,
                    "end": e.end,
                    "type": e.entity_type.value,
                    "text": e.text,
                }
                for e in entities
            ],
            "count": len(entities),
        }
    )


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or ""
    lower_name = filename.lower()
    if not lower_name.endswith((".pdf", ".docx", ".txt")):
        return jsonify({"error": "Supports PDF, DOCX, TXT"}), 400

    suffix = "." + lower_name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if lower_name.endswith(".pdf"):
            with pdfplumber.open(tmp_path) as pdf:
                plain = "\n".join(page.extract_text() or "" for page in pdf.pages)
            pages = len(pdf.pages)
        elif lower_name.endswith(".docx"):
            import docx

            doc = docx.Document(tmp_path)
            plain = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            pages = 1
        else:
            with open(tmp_path, "r", encoding="utf-8") as handle:
                plain = handle.read()
            pages = 1

        style = request.form.get("style", "placeholder")
        entities = redactor.detect(plain)
        redacted_text = redactor.redact(plain, replacement_style=style)
        return jsonify(
            {
                "filename": filename,
                "plain_text": plain,
                "markdown": plain,
                "redacted": redacted_text,
                "entities": [
                    {
                        "start": e.start,
                        "end": e.end,
                        "type": e.entity_type.value,
                        "text": e.text,
                    }
                    for e in entities
                ],
                "count": len(entities),
                "pages": pages,
            }
        )
    finally:
        os.unlink(tmp_path)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path: str):
    target = WEB_DIR / path
    if path and target.exists() and target.is_file():
        return send_from_directory(WEB_DIR, path)
    return send_from_directory(WEB_DIR, "index.html")


if __name__ == "__main__":
    print("Legal PII Redactor complete package")
    print(f"Mode: {'regex+ner' if USE_NER else 'regex-only'}")
    print(f"URL: http://127.0.0.1:{PORT}")
    try:
        from waitress import serve

        serve(app, host="0.0.0.0", port=PORT)
    except ImportError:
        app.run(host="0.0.0.0", port=PORT, debug=False)
