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

def load_model(weight_path: str = "plant_disease_resnet18_finetuned.pth"):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

model = load_model()


def build_advice_prompt(disease_class: str, confidence: float) -> str:
    return (
        "你是一位植物病害照護助理。請根據影像模型辨識結果，"
        "用繁體中文給一般使用者實用、保守、容易執行的建議。"
        "請不要聲稱這是絕對診斷，也不要提供危險或過量的農藥用法。\n\n"
        f"辨識結果：{disease_class}\n"
        f"模型信心值：{confidence:.1f}%\n\n"
        "請用 4 到 6 個短句或條列回覆，包含：\n"
        "1. 可能狀況\n"
        "2. 立即處理\n"
        "3. 隔離或修剪建議\n"
        "4. 澆水、通風或環境管理\n"
        "5. 何時應請教農業專家或園藝店"
    )


def extract_gemini_text(data: dict) -> str:
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "\n".join(part.get("text", "") for part in parts).strip()
    return text


def get_gemini_advice(disease_class: str, confidence: float) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_advice_prompt(disease_class, confidence)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Gemini API: {exc.reason}") from exc

    text = extract_gemini_text(data)
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


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

    if "healthy" in disease_class.lower():
        return jsonify({
            "advice": (
                "目前模型判斷葉片偏健康。建議持續保持適度日照、良好通風，"
                "避免葉面長時間潮濕，並每週觀察是否出現斑點、捲曲或變色。"
            )
        })

    try:
        suggestion = get_gemini_advice(disease_class, confidence)
    except RuntimeError as exc:
        app.logger.warning("Gemini advice failed: %s", exc)
        message = str(exc)
        if "GEMINI_API_KEY is not set" in message:
            error = "找不到 GEMINI_API_KEY，請確認 .env 在專案根目錄且 key 名稱正確。"
        elif "Cannot reach Gemini API" in message:
            error = "無法連線到 Gemini API，請確認網路、防火牆或代理伺服器設定。"
        elif "Gemini API error" in message:
            error = "Gemini API 回傳錯誤，請確認 API key、模型名稱與額度是否正常。"
        else:
            error = "暫時無法取得 Gemini 建議，請稍後再試。"
        response = {"error": error}
        if app.debug:
            response["detail"] = message[:500]
        return jsonify(response), 503

    return jsonify({"advice": suggestion})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
