const recordButton = document.querySelector("#record-button");
const recordLabel = document.querySelector("#record-label");
const statusText = document.querySelector("#status");
const recordMeta = document.querySelector(".record-meta");
const timerText = document.querySelector("#timer");
const transcript = document.querySelector("#transcript");
const resultMeta = document.querySelector("#result-meta");
const modelTime = document.querySelector("#model-time");

const maxSeconds = 90;
let state = "idle";
let context;
let stream;
let source;
let capture;
let chunks = [];
let startedAt = 0;
let timer;
let resolveFlush;

function setState(next, message) {
  state = next;
  recordButton.disabled = next === "working";
  recordButton.classList.toggle("is-recording", next === "recording");
  recordButton.setAttribute("aria-pressed", String(next === "recording"));
  recordMeta.classList.toggle("is-recording", next === "recording");
  recordMeta.classList.toggle("is-working", next === "working");
  recordLabel.textContent = next === "recording" ? "Stop and transcribe" : next === "working" ? "Transcribing…" : "Start speaking";
  statusText.textContent = message;
}

function showTimer() {
  const elapsed = Math.min(maxSeconds, Math.floor((performance.now() - startedAt) / 1000));
  timerText.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  if (elapsed >= maxSeconds) stopAndTranscribe();
}

function cleanupCapture() {
  clearInterval(timer);
  source?.disconnect();
  capture?.disconnect();
  stream?.getTracks().forEach((track) => track.stop());
  if (context && context.state !== "closed") context.close();
  context = stream = source = capture = null;
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    setState("idle", "This browser cannot capture audio here. Try a current version of Edge or Chrome.");
    return;
  }
  recordButton.disabled = true;
  statusText.textContent = "Waiting for microphone permission…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    context = new AudioContext();
    await context.audioWorklet.addModule("/mic-processor.js");
    source = context.createMediaStreamSource(stream);
    capture = new AudioWorkletNode(context, "mic-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    chunks = [];
    capture.port.onmessage = ({ data }) => {
      if (data.type === "chunk") chunks.push(new Float32Array(data.samples));
      if (data.type === "flushed") resolveFlush?.();
    };
    source.connect(capture);
    capture.connect(context.destination);
    await context.resume();
    startedAt = performance.now();
    timerText.textContent = "00:00";
    setState("recording", "Listening. Speak naturally, then stop to see the text.");
    recordButton.disabled = false;
    timer = setInterval(showTimer, 250);
  } catch (error) {
    cleanupCapture();
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    setState("idle", denied ? "Microphone access is blocked. Allow it for this localhost page, then try again." : "Could not start the microphone. Check that it is connected and try again.");
    recordButton.disabled = false;
  }
}

function wavBlob(parts, sampleRate) {
  const sampleCount = parts.reduce((sum, part) => sum + part.length, 0);
  const dataBytes = sampleCount * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeText = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = 44;
  for (const part of parts) {
    for (let i = 0; i < part.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, part[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function stopAndTranscribe() {
  if (state !== "recording" || !capture) return;
  clearInterval(timer);
  setState("working", "Preparing the recording locally…");
  const sampleRate = context.sampleRate;
  await new Promise((resolve) => {
    resolveFlush = resolve;
    capture.port.postMessage("flush");
  });
  resolveFlush = null;
  cleanupCapture();
  const audio = wavBlob(chunks, sampleRate);
  const duration = audio.size > 44 ? (audio.size - 44) / (sampleRate * 2) : 0;
  chunks = [];
  if (duration < 0.3) {
    setState("idle", "That recording was too short. Speak a phrase before stopping.");
    timerText.textContent = "00:00";
    return;
  }
  try {
    setState("working", "Transcribing on this device…");
    const response = await fetch("/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: audio });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Transcription failed. Try again.");
    transcript.value = result.text || "";
    resultMeta.textContent = `${result.seconds.toFixed(1)} s audio`;
    modelTime.textContent = `Recognized in ${result.asr_seconds.toFixed(2)} s`;
    setState("idle", result.text ? "Done. You can edit the transcription below." : "No words came through. Try speaking a little closer to the microphone.");
    transcript.focus();
  } catch (error) {
    setState("idle", error.message || "Local transcription failed. Try again.");
  } finally {
    chunks = [];
  }
}

recordButton.addEventListener("click", () => {
  if (state === "recording") stopAndTranscribe();
  else if (state === "idle") startRecording();
});
