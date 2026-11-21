"use strict";
const CSRF = document.querySelector('meta[name="csrf-token"]').content;
const $ = (id) => document.getElementById(id);

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await res.json(); } catch (e) { data = { ok: false, error: "Server returned an unexpected response (" + res.status + ")." }; }
  return data;
}

let stream = null;
async function startCamera(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot access a camera here. Open http://127.0.0.1:5000 in Chrome or Edge.");
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    const map = {
      NotAllowedError: "Camera permission was denied. Allow camera access in the browser's address bar and try again.",
      NotFoundError: "No camera was found on this computer.",
      NotReadableError: "The camera is in use by another app or tab. Close it and try again.",
    };
    throw new Error(map[e.name] || "Could not start the camera: " + e.message);
  }
  video.srcObject = stream;
  await video.play();
}
function stopCamera(video) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  if (video) video.srcObject = null;
}
function grabFrame(video, canvas, maxW, quality) {
  const scale = Math.min(1, maxW / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

/* ---------- training (students page) ---------- */
function initTrain() {
  const btn = $("train-btn"), out = $("train-result");
  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Training…";
    const r = await postJSON("/api/train", { equalize: $("eq").checked });
    out.classList.remove("hidden");
    out.replaceChildren();
    const p = document.createElement("p");
    if (r.ok) {
      p.textContent = `Trained on ${r.num_images} images of ${r.num_identities} identities (equalisation ${r.equalize ? "on" : "off"}).`;
      out.append(p);
      (r.warnings || []).forEach((w) => { const q = document.createElement("p"); q.className = "warn"; q.textContent = w; out.append(q); });
    } else { p.className = "warn"; p.textContent = r.error; out.append(p); }
    btn.disabled = false; btn.textContent = "Train model";
  });
}

/* ---------- capture (student page) ---------- */
function initCapture(sid) {
  const video = $("video"), work = $("work"), msg = $("cap-msg");
  const on = $("cam-on"), off = $("cam-off"), snap = $("snap"), burst = $("burst");
  const setBusy = (b) => { snap.disabled = b; burst.disabled = b; };
  async function capture() {
    const r = await postJSON(`/api/students/${sid}/capture`, { image: grabFrame(video, work, 960, 0.92) });
    if (r.ok) {
      $("count").textContent = r.count;
      const nt = $("no-thumbs"); if (nt) nt.remove();
      const img = document.createElement("img");
      img.src = r.url; img.width = 100; img.height = 100; img.alt = "Captured face";
      img.style.cssText = "border-radius:6px;border:1px solid var(--line);object-fit:cover";
      $("thumbs").prepend(img);
      msg.textContent = `Saved image ${r.count}.`;
    } else { msg.textContent = r.error; }
    return r.ok;
  }
  on.addEventListener("click", async () => {
    try { await startCamera(video); on.disabled = true; off.disabled = false; setBusy(false); msg.textContent = "Camera ready."; }
    catch (e) { msg.textContent = e.message; }
  });
  off.addEventListener("click", () => { stopCamera(video); on.disabled = false; off.disabled = true; setBusy(true); });
  snap.addEventListener("click", async () => { setBusy(true); await capture(); setBusy(false); });
  burst.addEventListener("click", async () => {
    setBusy(true);
    let saved = 0;
    for (let i = 0; i < 25 && saved < 10; i++) {
      if (await capture()) saved++;
      await new Promise((r) => setTimeout(r, 700));
    }
    msg.textContent = `Auto-capture finished: ${saved} image(s) saved. Reload to review them.`;
    setBusy(false);
  });
  window.addEventListener("beforeunload", () => stopCamera(video));
}

/* ---------- live attendance ---------- */
function initLive() {
  const video = $("video"), overlay = $("overlay"), work = $("work"), msg = $("live-msg");
  const start = $("start"), stop = $("stop"), state = $("state"), log = $("log");
  let running = false, sessionId = "";
  const colour = { marked: "#1f9d55", already_marked: "#1f9d55", confirming: "#d19a00", unknown: "#d64545" };
  const label = { marked: "Present", already_marked: "Already marked", confirming: "Confirming…", unknown: "" };

  function draw(res) {
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    const ctx = overlay.getContext("2d"), k = overlay.width / res.frame[0];
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = 3; ctx.font = "bold 18px Segoe UI, sans-serif"; ctx.textBaseline = "bottom";
    res.faces.forEach((f) => {
      const [x, y, w, h] = f.box.map((v) => v * k);
      const mx = overlay.width - (x + w); // video element is mirrored, canvas is not
      ctx.strokeStyle = colour[f.status]; ctx.strokeRect(mx, y, w, h);
      const text = (f.status === "unknown" ? "Unknown" : f.name) + (label[f.status] ? " · " + label[f.status] : "");
      const tw = ctx.measureText(text).width + 10;
      ctx.fillStyle = colour[f.status]; ctx.fillRect(mx, Math.max(y - 26, 0), tw, 26);
      ctx.fillStyle = "#fff"; ctx.fillText(text, mx + 5, Math.max(y - 2, 26));
    });
  }
  function addLog(name) {
    if (log.firstChild && log.firstChild.classList && log.firstChild.classList.contains("muted")) log.replaceChildren();
    const li = document.createElement("li");
    li.textContent = `${new Date().toLocaleTimeString()} — ${name} marked present`;
    log.prepend(li);
  }
  async function loop() {
    while (running) {
      const t0 = performance.now();
      const res = await postJSON("/api/recognize", { image: grabFrame(video, work, 640, 0.8), session_id: sessionId, mark: true });
      if (!running) break;
      if (!res.ok) { msg.textContent = res.error; stopAll(); return; }
      msg.textContent = res.faces.length ? `${res.faces.length} face(s) in view.` : "No face detected.";
      draw(res);
      res.marked.forEach(addLog);
      await new Promise((r) => setTimeout(r, Math.max(0, 450 - (performance.now() - t0))));
    }
  }
  function stopAll() {
    running = false; stopCamera(video);
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
    start.disabled = false; stop.disabled = true; state.textContent = "Stopped";
  }
  start.addEventListener("click", async () => {
    start.disabled = true;
    try { await startCamera(video); } catch (e) { msg.textContent = e.message; start.disabled = false; return; }
    sessionId = Math.random().toString(36).slice(2, 12);
    running = true; stop.disabled = false; state.textContent = "Recognising"; msg.textContent = "";
    loop();
  });
  stop.addEventListener("click", stopAll);
  window.addEventListener("beforeunload", () => stopCamera(video));
}

/* ---------- evaluation ---------- */
function initEvaluation() {
  const msg = $("eval-msg");
  document.querySelectorAll("[data-eval]").forEach((b) => b.addEventListener("click", async () => {
    msg.textContent = "Running…";
    const r = await postJSON("/api/evaluation/run", { kind: b.dataset.eval, threshold: $("thr").value });
    if (!r.ok) { msg.textContent = r.error; return; }
    msg.textContent = "Done. Reloading…";
    location.reload();
  }));
  const video = $("video"), work = $("work"), cap = $("cap-msg"), on = $("cam-on"), snap = $("snap-test");
  on.addEventListener("click", async () => {
    try { await startCamera(video); snap.disabled = false; on.disabled = true; } catch (e) { cap.textContent = e.message; }
  });
  snap.addEventListener("click", () => {
    grabFrame(video, work, 1600, 0.95);
    work.toBlob((blob) => {
      const dt = new DataTransfer();
      dt.items.add(new File([blob], "webcam.jpg", { type: "image/jpeg" }));
      $("test-file").files = dt.files;
      cap.textContent = "Frame captured into the image field. Enter the people and save.";
    }, "image/jpeg", 0.95);
  });
  window.addEventListener("beforeunload", () => stopCamera(video));
}
