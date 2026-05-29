import io
import json
import os
import torch
import torchvision.transforms as transforms
import urllib.error
import urllib.request
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(BASE_DIR, ".env"))

CLASS_NAMES = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]

MEAN = [0.4737, 0.4874, 0.4141]
STD  = [0.1893, 0.1750, 0.2034]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO   = "iamhenryhuang/plant_disease_resnet18_finetuned"
HF_FILE   = "plant_disease_resnet18_finetuned.pth"
WEIGHT_PATH = os.path.join(os.environ.get("WEIGHTS_DIR", BASE_DIR), HF_FILE)

def _ensure_weights() -> None:
    if os.path.exists(WEIGHT_PATH):
        return
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
    print(f"權重檔不存在，從 HuggingFace 下載中：{url}")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=120) as resp, open(WEIGHT_PATH, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        print("下載完成。")
    except Exception as exc:
        if os.path.exists(WEIGHT_PATH):
            os.remove(WEIGHT_PATH)
        raise RuntimeError(f"無法下載權重檔：{exc}") from exc


def load_model() -> torch.nn.Module:
    _ensure_weights()
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

model = load_model()


def build_advice_prompt(disease_class: str, confidence: float) -> str:
    return (
        "你是一位植物病蟲害照護顧問。請用繁體中文回答，語氣清楚、實用、適合一般種植者。\n"
        "模型已辨識出植物可能的狀態，請不要只說明遇到什麼問題；一定要提供可執行的解決方法。\n\n"
        f"辨識結果：{disease_class}\n"
        f"模型信心分數：{confidence:.1f}%\n\n"
        "請嚴格依照下面格式回答，每一段都要有完整內容。\n"
        "【重要格式規定】不可使用 Markdown 符號（#、*、-、` 等），只用純文字。\n"
        "每個段落標題用「■ 」開頭，子項目用「• 」開頭，段落之間空一行。\n\n"
        "■ 可能問題\n"
        "（簡短說明這個病害或健康狀態代表什麼）\n\n"
        "■ 立即處理\n"
        "• 步驟一\n"
        "• 步驟二\n"
        "• 步驟三\n\n"
        "■ 防治方式\n"
        "（具體治療或防治建議，包含非農藥方法；若提到藥劑請提醒依當地標示與安全規範使用）\n\n"
        "■ 後續照護\n"
        "（說明接下來 1 到 2 週的觀察、澆水、通風、日照與施肥建議）\n\n"
        "■ 何時需要求助\n"
        "（說明什麼情況下應詢問農業專家、園藝店或當地農業單位）\n\n"
        "若辨識結果是 Healthy，請依上述格式提供維持健康的照護建議，不要硬說有病。\n"
        "若信心分數偏低，請在可能問題段落提醒使用者再拍攝清楚葉片正反面、莖部與整株照片確認。\n"
        "請避免保證一定治癒，也不要提供危險或過量用藥建議。"
    )


def _normalize_advice(text: str) -> str:
    import re
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        # ATX headings: ## Foo  →  ■ Foo
        s = re.sub(r"^#{1,6}\s+", "■ ", s)
        # bold **text** or __text__  →  text (kept as-is, front-end will bold section titles)
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"__(.+?)__", r"\1", s)
        # italic *text* or _text_  →  text
        s = re.sub(r"\*(.+?)\*", r"\1", s)
        s = re.sub(r"_(.+?)_", r"\1", s)
        # unordered list markers: - / * at line start  →  •
        s = re.sub(r"^[-*]\s+", "• ", s)
        # ordered list: 1. / 2. etc  →  keep number with dot, prepend •
        s = re.sub(r"^\d+\.\s+", lambda m: "• ", s)
        # section headings written as "1. 可能問題：..." (numbered Chinese sections)
        # detect pattern: optional bullet + digit + dot|、 + Chinese label + colon
        heading = re.match(r"^[•\s]*\d+[\.、]\s*(.+)", s)
        if heading:
            s = f"■ {heading.group(1)}"
        # horizontal rules
        if re.match(r"^[-*_]{3,}$", s):
            s = ""
        out.append(s)
    return "\n".join(out).strip()


def extract_groq_text(data: dict) -> str:
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    ).strip()
    return _normalize_advice(text)


def get_groq_advice(disease_class: str, confidence: float) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": build_advice_prompt(disease_class, confidence)}],
        "temperature": 0.4,
        "max_tokens": 2048,
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "plant-disease-app/1.0",
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = extract_groq_text(data)
        if not text:
            raise RuntimeError("Groq returned an empty response")
        return text
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Groq API: {exc.reason}") from exc


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open image"}), 400

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3 = torch.topk(probs, k=3)
    results = [
        {
            "class": CLASS_NAMES[idx.item()],
            "confidence": round(prob.item() * 100, 2),
        }
        for prob, idx in zip(top3.values, top3.indices)
    ]

    return jsonify({"predictions": results})


@app.route("/advice", methods=["POST"])
def advice():
    data = request.get_json(silent=True) or {}
    disease_class = data.get("class", "")
    confidence = data.get("confidence", 0)

    if disease_class not in CLASS_NAMES:
        return jsonify({"error": "Unknown disease class"}), 400

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid confidence"}), 400

    try:
        suggestion = get_groq_advice(disease_class, confidence)
    except RuntimeError as exc:
        app.logger.warning("Groq advice failed: %s", exc)
        message = str(exc)
        if "GROQ_API_KEY is not set" in message:
            error = "找不到 GROQ_API_KEY，請確認 .env 在專案根目錄且 key 名稱正確。"
        elif "Cannot reach Groq API" in message:
            error = "無法連線到 Groq API，請確認網路、防火牆或代理伺服器設定。"
        elif "Groq API error" in message:
            error = "Groq API 回傳錯誤，請確認 API key 與額度是否正常。"
        else:
            error = "暫時無法取得建議，請稍後再試。"
        response = {"error": error}
        if app.debug:
            response["detail"] = message[:500]
        return jsonify(response), 503

    return jsonify({"advice": suggestion})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
