const dzWrap      = document.getElementById("dz-wrap");
const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const prevWrap    = document.getElementById("preview-wrap");
const prevImg     = document.getElementById("preview-img");
const fileName    = document.getElementById("file-name");
const analyzeBtn  = document.getElementById("analyze-btn");
const resetBtn    = document.getElementById("reset-btn");
const spinner     = document.getElementById("spinner");
const empty       = document.getElementById("empty");
const resList     = document.getElementById("results-list");
const errMsg      = document.getElementById("error-msg");
const errTxt      = document.getElementById("error-text");
const verdict     = document.getElementById("verdict");
const vTag        = document.getElementById("v-tag");
const vTagLbl     = document.getElementById("v-tag-label");
const vName       = document.getElementById("v-name");
const vBarFill    = document.getElementById("v-bar-fill");
const vPct        = document.getElementById("v-pct");
const vZh         = document.getElementById("v-zh");
const hdStatus    = document.getElementById("hd-status");
const hdLabel     = document.getElementById("hd-label");
const adviceEmpty = document.getElementById("advice-empty");
const adviceBody  = document.getElementById("advice-body");
const adviceNote  = document.getElementById("advice-note");
const adviceStat  = document.getElementById("advice-status");
const providerSelect = document.getElementById("provider-select");
const adviceTitle    = document.getElementById("advice-title");

const ZH = {
  "Pepper Bell - Bacterial Spot":                   "甜椒 — 細菌性斑點病",
  "Pepper Bell - Healthy":                          "甜椒 — 健康",
  "Potato - Early Blight":                          "馬鈴薯 — 早疫病",
  "Potato - Late Blight":                           "馬鈴薯 — 晚疫病",
  "Potato - Healthy":                               "馬鈴薯 — 健康",
  "Tomato - Bacterial Spot":                        "番茄 — 細菌性斑點病",
  "Tomato - Early Blight":                          "番茄 — 早疫病",
  "Tomato - Late Blight":                           "番茄 — 晚疫病",
  "Tomato - Leaf Mold":                             "番茄 — 葉黴病",
  "Tomato - Septoria Leaf Spot":                    "番茄 — 葉枯病",
  "Tomato - Spider Mites (Two-Spotted Spider Mite)":"番茄 — 二點葉蟎",
  "Tomato - Target Spot":                           "番茄 — 靶斑病",
  "Tomato - Yellow Leaf Curl Virus":                "番茄 — 黃化捲葉病毒",
  "Tomato - Mosaic Virus":                          "番茄 — 嵌紋病毒",
  "Tomato - Healthy":                               "番茄 — 健康",
};

function zh(cls) { return ZH[cls] || ""; }

let file = null;

function setStatus(txt, s = "") { hdLabel.textContent = txt; hdStatus.dataset.s = s; }
function isHealthy(c) { return c.toLowerCase().includes("healthy"); }

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("over"); });
dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("over"); if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) load(fileInput.files[0]); });

function load(f) {
  if (!f.type.startsWith("image/")) { showErr("請上傳圖片檔案。"); return; }
  file = f;
  clearErr(); clearResults();
  const r = new FileReader();
  r.onload = e => {
    prevImg.src = e.target.result;
    fileName.textContent = f.name;
    dzWrap.style.display = "none";
    prevWrap.classList.add("on");
    setStatus("影像已載入", "ready");
  };
  r.readAsDataURL(f);
}

analyzeBtn.addEventListener("click", async () => {
  if (!file) return;
  analyzeBtn.disabled = true;
  spinner.classList.add("on");
  clearResults(); clearErr();
  setStatus("推論中…", "busy");

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res  = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || data.error) { showErr(data.error || "辨識失敗，請重新嘗試。"); return; }
    render(data.predictions);
    await loadAdvice(data.predictions[0]);
  } catch {
    showErr("無法連線至辨識服務。");
  } finally {
    spinner.classList.remove("on");
    analyzeBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  file = null; fileInput.value = "";
  prevImg.src = ""; fileName.textContent = "";
  dzWrap.style.display = "";
  prevWrap.classList.remove("on");
  clearResults(); clearErr();
  setStatus("等待上傳", "");
});

function render(preds) {
  const top = preds[0];
  const ok  = isHealthy(top.class);
  const pct = top.confidence.toFixed(1);

  vTag.className      = "v-tag fi " + (ok ? "ok" : "diseased");
  vTagLbl.textContent = ok ? "健康" : "疾病偵測";
  vName.textContent   = top.class;
  vZh.textContent     = zh(top.class);
  vPct.textContent    = pct + "%";
  verdict.style.display = "flex";
  verdict.classList.add("fi");

  requestAnimationFrame(() => { vBarFill.style.width = pct + "%"; });

  resList.innerHTML = "";
  preds.slice(0, 3).forEach((item, i) => {
    const p  = item.confidence.toFixed(1);
    const el = document.createElement("div");
    el.className = "fi";
    el.style.animationDelay = `${i * 40}ms`;
    el.innerHTML = `
      <div class="rrow${i === 0 ? " r1" : ""}">
        <span class="rnum">${i + 1}</span>
        <span class="rname">${item.class}<span class="rname-zh">${zh(item.class)}</span></span>
        <span class="rpct">${p}%</span>
      </div>
      <div class="rbar"><div class="rfill" data-w="${p}"></div></div>`;
    resList.appendChild(el);
  });

  empty.style.display = "none";
  resList.style.display = "block";
  requestAnimationFrame(() => {
    resList.querySelectorAll(".rfill").forEach(b => { b.style.width = b.dataset.w + "%"; });
  });
}

function mdToHtml(text) {
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const lines = text.split("\n");
  let html = "";
  for (const raw of lines) {
    const line = esc(raw.trim());
    if (!line) { html += "<br>"; continue; }
    if (line.startsWith("■ ")) {
      html += `<p style="font-weight:700;color:var(--p-600);margin:12px 0 4px">${line.slice(2)}</p>`;
    } else if (line.startsWith("• ")) {
      html += `<p style="margin:2px 0 2px 12px">• ${line.slice(2)}</p>`;
    } else {
      html += `<p style="margin:2px 0">${line}</p>`;
    }
  }
  return html;
}

function showAdviceContent(text) {
  adviceEmpty.style.display = "none";
  adviceBody.style.display  = "block";
  adviceNote.style.display  = "block";
  adviceBody.innerHTML      = mdToHtml(text);
  adviceBody.classList.add("fi");
}

function resetAdvice() {
  adviceEmpty.style.display = "";
  adviceBody.style.display  = "none";
  adviceNote.style.display  = "none";
  adviceBody.innerHTML      = "";
  adviceBody.classList.remove("fi");
  adviceStat.classList.remove("on");
}

providerSelect.addEventListener("change", () => {
  adviceTitle.textContent = "LLM 照護建議";
});

async function loadAdvice(top) {
  adviceEmpty.style.display = "none";
  adviceStat.classList.add("on");
  setStatus("產生建議中…", "busy");

  const provider = providerSelect.value;

  try {
    const res  = await fetch("/advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class: top.class, confidence: top.confidence, provider }),
    });
    const data = await res.json();
    showAdviceContent(!res.ok || data.error ? (data.error || "暫時無法產生照護建議。") : data.advice);
    setStatus("辨識完成", "ready");
  } catch {
    showAdviceContent("無法連線至建議服務，請稍後再試。");
    setStatus("辨識完成", "ready");
  } finally {
    adviceStat.classList.remove("on");
  }
}

function clearResults() {
  resList.style.display = "none";
  empty.style.display   = "";
  verdict.style.display = "none";
  resList.innerHTML     = "";
  resetAdvice();
  vBarFill.style.width  = "0";
}

function showErr(msg) { errTxt.textContent = msg; errMsg.classList.add("on"); setStatus("發生錯誤", "error"); }
function clearErr()   { errMsg.classList.remove("on"); errTxt.textContent = ""; }
