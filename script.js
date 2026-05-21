import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// ── DOM ──────────────────────────────────────────────
const video             = document.getElementById("webcam");
const canvas            = document.getElementById("outputCanvas");
const canvasCtx         = canvas.getContext("2d");
const startBtn          = document.getElementById("startBtn");
const addWordBtn        = document.getElementById("addWordBtn");
const clearSentenceBtn  = document.getElementById("clearSentenceBtn");
const speakSentenceBtn  = document.getElementById("speakSentenceBtn");
const datasetStatus     = document.getElementById("datasetStatus");
const cameraStatus      = document.getElementById("cameraStatus");
const handStatus        = document.getElementById("handStatus");
const runStatus         = document.getElementById("runStatus");
const currentPrediction = document.getElementById("currentPrediction");
const predictionScore   = document.getElementById("predictionScore");
const holdStatus        = document.getElementById("holdStatus");
const holdBarFill       = document.getElementById("holdBarFill");
const sentenceOutput    = document.getElementById("sentenceOutput");
const labelChips        = document.getElementById("labelChips");

// ── 설정 ─────────────────────────────────────────────
let   LIVE_BUFFER_MAX    = 100; // metadata.json에서 덮어씀
const ML_FEATURE_DIM     = 90;
const MIN_SCORE_SHOW     = 0.70;
const MIN_SCORE_LOCK     = 0.75;
const MIN_MARGIN         = 0.20;
const MIN_GESTURE_FRAMES = 15;   // 최소 녹화 프레임 (너무 짧으면 무시)
const NO_HAND_END_FRAMES = 8;    // 손이 N프레임 없으면 제스처 종료로 판단
const COOLDOWN_MS        = 1500; // 인식 후 다음 제스처까지 대기 시간
const LOCK_HOLD_MS       = 700;  // 신뢰도 유지 시간 (ms)
const FACE_KEY_POINTS    = [4, 10, 13, 33, 152, 234, 263, 454];

// ── 상태 ─────────────────────────────────────────────
let handLandmarker      = null;
let faceLandmarker      = null;
let drawingUtils        = null;
let mlModel             = null;
let webcamRunning       = false;
let lastVideoTime       = -1;
let appStarted          = false;
let autoAddEnabled      = true;

let labelsConfig        = [];
let targetLabels        = [];
let sentenceWords       = [];
let latestDetectedHands = [];

// 제스처 구간 감지 상태
let gestureState    = 'waiting';
let gestureBuffer   = [];
let noHandCount     = 0;
let lastResult      = null;
let liveInferCount  = 0;  // 실시간 추론 프레임 카운터
let lockStartTime   = 0;  // 신뢰도 유지 시작 시각 (ms)
let lastFaceResults = null;
let lastHandResults = null;
let faceFrameCount  = 0;
let handFrameCount  = 0;

// MediaPipe 추론 전용 640×480 캔버스 (표시용 비디오 해상도와 분리)
const inferCanvas = document.createElement('canvas');
inferCanvas.width  = 640;
inferCanvas.height = 480;
const inferCtx = inferCanvas.getContext('2d');

// ── 1. labels.json 로드 ───────────────────────────────
async function loadLabels() {
  try {
    const res = await fetch("./labels.json");
    if (!res.ok) throw new Error();
    const json = await res.json();
    labelsConfig = json.labels || [];
    targetLabels = labelsConfig.map(l => l.id);
    renderLabelChips();
  } catch {
    labelsConfig = [];
    targetLabels = [];
  }
}

function renderLabelChips() {
  if (!labelChips) return;
  labelChips.innerHTML = labelsConfig
    .filter(l => l.id !== "기타")
    .map(l => `<span class="chip">${l.korean}</span>`)
    .join("");
}

// ── 2. ML 모델 로드 ───────────────────────────────────
async function loadMLModel() {
  try {
    datasetStatus.textContent = "ML 모델 로딩 중...";
    await tf.setBackend('webgl');
    await tf.ready();
    const meta = await fetch("./model/metadata.json").then(r => r.json()).catch(() => null);
    if (meta && meta.max_sequence_length) LIVE_BUFFER_MAX = meta.max_sequence_length;
    mlModel = await tf.loadLayersModel("./model/tfjs_model/model.json");
    datasetStatus.textContent = "✅ ML 모델 로딩 완료";
  } catch (e) {
    datasetStatus.textContent = "❌ " + e.message;
    console.error(e);
  }
}

// ── 3. MediaPipe 초기화 ───────────────────────────────
async function createLandmarkers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });
  drawingUtils = new DrawingUtils(canvasCtx);
}

// ── 4. 카메라 ─────────────────────────────────────────
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = () => resolve(); });
    await video.play();
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    webcamRunning = true;
    cameraStatus.textContent = "카메라 연결 완료";
  } catch (e) {
    cameraStatus.textContent = "카메라 연결 실패";
    console.error(e);
  }
}

// ── 5. 랜드마크 그리기 ────────────────────────────────
function drawResults(handResults, faceResults) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.translate(canvas.width, 0);
  canvasCtx.scale(-1, 1);

  if (handResults && handResults.landmarks) {
    for (const lm of handResults.landmarks) {
      drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS,
        { lineWidth: 3, color: "#4f8cff" });
      drawingUtils.drawLandmarks(lm, { radius: 4, color: "#7c3aed" });
    }
  }

  if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
    const faceLM = faceResults.faceLandmarks[0];
    for (const idx of FACE_KEY_POINTS) {
      const pt = faceLM[idx];
      if (!pt) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, Math.PI * 2);
      canvasCtx.fillStyle = "rgba(255, 200, 0, 0.85)";
      canvasCtx.fill();
    }
  }

  canvasCtx.restore();
}

// ── 6. 메인 루프 ──────────────────────────────────────
function predictWebcam() {
  if (!appStarted || !webcamRunning || !handLandmarker) return;
  const now = performance.now();

  // 그리기는 항상 (캐시된 결과 사용) → 점이 끊기지 않음
  drawResults(lastHandResults, lastFaceResults);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // 추론 캔버스에 현재 프레임 축소 복사
    inferCtx.drawImage(video, 0, 0, 640, 480);

    // 손 감지: 3프레임마다
    handFrameCount++;
    if (handFrameCount % 3 === 0) {
      lastHandResults = handLandmarker.detectForVideo(inferCanvas, now);
      const prev = latestDetectedHands.length;
      latestDetectedHands = cloneHands(lastHandResults);
      const cur = latestDetectedHands.length;
      if (prev !== cur) {
        handStatus.textContent = cur > 0 ? `손 감지됨: ${cur}개` : "손 감지되지 않음";
      }
    }

    // 얼굴 감지: 9프레임마다
    faceFrameCount++;
    if (faceFrameCount % 9 === 0) {
      lastFaceResults = faceLandmarker.detectForVideo(inferCanvas, now);
    }

    updateGestureBuffer(lastHandResults, lastFaceResults);
  }

  requestAnimationFrame(predictWebcam);
}

// ── 7. 손 데이터 복제 ─────────────────────────────────
function cloneHands(handResults) {
  if (!handResults || !handResults.landmarks) return [];
  return handResults.landmarks.map((lm, i) => ({
    handedness: handResults.handedness?.[i]?.[0]?.categoryName || "Unknown",
    landmarks: lm.map(p => ({ x: p.x, y: p.y, z: p.z }))
  }));
}

// ── 8. 특징 추출 (상대 좌표 + 손바닥 법선 벡터) ──────
function extractMLFrame(handResults, faceResults) {
  const features = [];

  // 손: 손목(landmark 0) 기준 상대 좌표 × 21개 = 63
  if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
    const lm    = handResults.landmarks[0];
    const wrist = lm[0];
    for (const pt of lm) {
      features.push(pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z);
    }

    // 손바닥 법선 벡터 (palm normal): cross(wrist→검지MCP, wrist→소지MCP) 정규화 = 3
    // 손바닥이 향하는 방향을 직접 인코딩해서 '나'/'너' 구분력 향상
    const idxMcp  = lm[5];
    const pinkMcp = lm[17];
    const v1x = idxMcp.x  - wrist.x,  v1y = idxMcp.y  - wrist.y,  v1z = idxMcp.z  - wrist.z;
    const v2x = pinkMcp.x - wrist.x,  v2y = pinkMcp.y - wrist.y,  v2z = pinkMcp.z - wrist.z;
    const nx = v1y * v2z - v1z * v2y;
    const ny = v1z * v2x - v1x * v2z;
    const nz = v1x * v2y - v1y * v2x;
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen > 1e-6) features.push(nx / nLen, ny / nLen, nz / nLen);
    else             features.push(0, 0, 0);
  } else {
    for (let i = 0; i < 66; i++) features.push(0);
  }

  // 얼굴: 코끝(첫 번째 키포인트) 기준 상대 좌표 × 8개 = 24
  if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
    const faceLM = faceResults.faceLandmarks[0];
    const nose   = faceLM[FACE_KEY_POINTS[0]];
    for (const idx of FACE_KEY_POINTS) {
      const pt = faceLM[idx];
      features.push(
        pt ? pt.x - nose.x : 0,
        pt ? pt.y - nose.y : 0,
        pt ? pt.z - nose.z : 0
      );
    }
  } else {
    for (let i = 0; i < 24; i++) features.push(0);
  }

  return features; // 63 + 3 + 24 = 90
}

// ── 9. 제스처 구간 감지 ───────────────────────────────
// 손 나타남 → 녹화 → 손 사라짐 → 분류 (학습 방식과 동일한 구조)
function updateGestureBuffer(handResults, faceResults) {
  if (gestureState === 'cooldown') return;

  const hasHand = latestDetectedHands.length > 0;

  // ── WAITING: 손 나타나면 녹화 시작
  if (gestureState === 'waiting') {
    if (hasHand) {
      gestureState   = 'recording';
      gestureBuffer  = [];
      noHandCount    = 0;
      liveInferCount = 0;
      lockStartTime  = 0;
      holdStatus.textContent        = "동작 중...";
      holdBarFill.style.width       = "0%";
      currentPrediction.textContent = "—";
      predictionScore.textContent   = "";
    }
    return;
  }

  // ── RECORDING: 프레임 수집
  if (gestureState === 'recording') {
    if (hasHand) {
      noHandCount = 0;
      gestureBuffer.push(extractMLFrame(handResults, faceResults));
      if (gestureBuffer.length > LIVE_BUFFER_MAX) gestureBuffer.shift();

      // 5프레임마다 추론 (매 프레임 추론하면 너무 느림)
      liveInferCount++;
      if (liveInferCount % 8 === 0 && gestureBuffer.length >= MIN_GESTURE_FRAMES) {
        const live = runMLInference(gestureBuffer);
        const t = performance.now();
        if (live && live.score >= MIN_SCORE_SHOW && live.label !== "기타") {
          const found = labelsConfig.find(l => l.id === live.label);
          currentPrediction.textContent = found ? found.korean : live.label;
          predictionScore.textContent   = `신뢰도: ${(live.score * 100).toFixed(1)}%`;
          holdBarFill.style.width       = `${(live.score * 100).toFixed(0)}%`;

          if (live.score >= MIN_SCORE_LOCK) {
            if (lockStartTime === 0) lockStartTime = t;
            const elapsed = t - lockStartTime;
            const remaining = Math.max(0, LOCK_HOLD_MS - elapsed);
            holdStatus.textContent = remaining > 50
              ? `동작 유지... (${(remaining / 1000).toFixed(1)}s)`
              : "확정!";
            if (elapsed >= LOCK_HOLD_MS) {
              classifyGesture();
              return;
            }
          } else {
            lockStartTime = 0;
            holdStatus.textContent = "손을 내리면 확정돼요";
          }
        } else {
          lockStartTime = 0;
          currentPrediction.textContent = "...";
          predictionScore.textContent   = "";
          holdStatus.textContent        = "동작 중...";
          holdBarFill.style.width       = "0%";
        }
      }
    } else {
      noHandCount++;
      lockStartTime = 0;
      if (noHandCount >= NO_HAND_END_FRAMES) {
        if (gestureBuffer.length >= MIN_GESTURE_FRAMES) {
          classifyGesture();
        } else {
          resetGesture();
          holdStatus.textContent = "너무 짧습니다. 다시 해주세요";
        }
      }
    }
  }
}

// ── 10. 제스처 분류 ───────────────────────────────────
function classifyGesture() {
  const best = runMLInference(gestureBuffer);

  gestureState  = 'cooldown';
  gestureBuffer = [];

  if (!best || best.score < MIN_SCORE_SHOW || best.label === "기타") {
    currentPrediction.textContent = "...";
    predictionScore.textContent   = "";
    holdStatus.textContent        = "인식 실패 — 다시 해주세요";
    holdBarFill.style.width       = "0%";
    lastResult = null;
    setTimeout(resetGesture, COOLDOWN_MS);
    return;
  }

  const found = labelsConfig.find(l => l.id === best.label);
  const word  = found ? found.korean : best.label;

  currentPrediction.textContent = word;
  predictionScore.textContent   = `신뢰도: ${(best.score * 100).toFixed(1)}%`;
  holdBarFill.style.width       = "100%";
  lastResult = best;

  if (autoAddEnabled && best.score >= MIN_SCORE_LOCK) {
    addWordToSentence(best.label);
  } else if (best.score >= MIN_SCORE_LOCK) {
    holdStatus.textContent = `✅ 추가 가능: ${word} (버튼 누르세요)`;
  } else {
    holdStatus.textContent = `신뢰도 부족 (${(best.score * 100).toFixed(1)}%) — 다시 해주세요`;
  }

  setTimeout(resetGesture, COOLDOWN_MS);
}

// ── 11. ML 추론 ───────────────────────────────────────
function runMLInference(frames) {
  if (!mlModel || frames.length < MIN_GESTURE_FRAMES) return null;

  const data = new Float32Array(LIVE_BUFFER_MAX * ML_FEATURE_DIM);
  const len  = Math.min(frames.length, LIVE_BUFFER_MAX);
  for (let i = 0; i < len; i++) {
    for (let j = 0; j < ML_FEATURE_DIM; j++) {
      data[i * ML_FEATURE_DIM + j] = frames[i][j] || 0;
    }
  }

  const inputTensor  = tf.tensor3d(data, [1, LIVE_BUFFER_MAX, ML_FEATURE_DIM]);
  const outputTensor = mlModel.predict(inputTensor);
  const probs        = outputTensor.dataSync();
  inputTensor.dispose();
  outputTensor.dispose();

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  let secondScore = 0;
  for (let i = 0; i < probs.length; i++) {
    if (i !== bestIdx && probs[i] > secondScore) secondScore = probs[i];
  }
  if (probs[bestIdx] - secondScore < MIN_MARGIN) return null;

  return { label: targetLabels[bestIdx], score: probs[bestIdx] };
}

// ── 12. 단어 추가 ─────────────────────────────────────
function addWordToSentence(labelId) {
  const found = labelsConfig.find(l => l.id === labelId);
  const word  = found ? found.korean : labelId;
  sentenceWords.push(word);
  updateSentenceUI();
  holdStatus.textContent  = `✅ "${word}" 추가됨!`;
}

function resetGesture() {
  gestureState  = 'waiting';
  gestureBuffer = [];
  noHandCount   = 0;
  holdBarFill.style.width = "0%";
  if (appStarted) holdStatus.textContent = "손을 카메라에 보여주세요";
}

// ── 13. 수동 추가 (버튼) ──────────────────────────────
function addWord() {
  if (!lastResult || lastResult.label === "기타") {
    alert("아직 인식된 단어가 없습니다.");
    return;
  }
  addWordToSentence(lastResult.label);
  lastResult = null;
}

function clearSentence() { sentenceWords = []; updateSentenceUI(); }

function updateSentenceUI() {
  sentenceOutput.textContent = sentenceWords.length > 0
    ? sentenceWords.join(" ")
    : "아직 추가된 단어가 없습니다.";
}

function speakSentence() {
  if (sentenceWords.length === 0) { alert("읽을 문장이 없습니다."); return; }
  const ttsWords = sentenceWords.map(w => {
    const found = labelsConfig.find(l => l.korean === w);
    return found ? found.tts : w;
  });
  const utt = new SpeechSynthesisUtterance(ttsWords.join(" "));
  utt.lang = "ko-KR";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utt);
}

// ── 이벤트 ───────────────────────────────────────────
startBtn.addEventListener("click", () => {
  appStarted = !appStarted;
  resetGesture();
  lastResult = null;
  if (appStarted) {
    startBtn.textContent          = "⏹ 중지";
    runStatus.textContent         = "실시간 인식 중 (자동 추가 켜짐)";
    currentPrediction.textContent = "—";
    predictionScore.textContent   = "";
    predictWebcam();
  } else {
    startBtn.textContent          = "▶ 시작";
    runStatus.textContent         = "대기 중";
    currentPrediction.textContent = "—";
    predictionScore.textContent   = "";
    holdStatus.textContent        = "안정화 대기 중";
    holdBarFill.style.width       = "0%";
  }
});

addWordBtn.addEventListener("click", addWord);
addWordBtn.addEventListener("dblclick", () => {
  autoAddEnabled = !autoAddEnabled;
  addWordBtn.textContent = autoAddEnabled ? "+ 단어 추가" : "+ 수동 추가";
  runStatus.textContent  = autoAddEnabled
    ? "실시간 인식 중 (자동 추가 켜짐)"
    : "실시간 인식 중 (자동 추가 꺼짐)";
});

clearSentenceBtn.addEventListener("click", clearSentence);
speakSentenceBtn.addEventListener("click", speakSentence);

// ── 앱 시작 ───────────────────────────────────────────
async function initApp() {
  await loadLabels();
  await loadMLModel();
  cameraStatus.textContent = "MediaPipe 준비 중...";
  await createLandmarkers();
  await setupCamera();
  runStatus.textContent = "대기 중 — 시작 버튼을 눌러주세요";
}

initApp();
