import {
  FaceLandmarker,
  FilesetResolver,
  ObjectDetector,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

// ==========================================
// APP STATE
// ==========================================
let faceLandmarker = null;
let objectDetector = null;
let handLandmarker = null;
let webcamRunning = false;
let stream = null;
let lastVideoTime = -1;

// Tracking state variables
let isCalibrated = false;
let isCalibrating = false;
let calibrationFrames = [];
let calibrationStartTimestamp = 0;
const CALIBRATION_DURATION_MS = 2000;
let baselineGaze = { x: 0, y: 0 };

// Gaze thresholds
const YAW_THRESHOLD = 0.06; // Lowered to bypass vertical checks even for very small horizontal head rotations
const PITCH_THRESHOLD = 0.20; // Increased so small upward movements or crosstalk don't trigger warning
const PITCH_DOWN_THRESHOLD = 0.45; // Vertical movement (looking DOWN - allows reading book)

// Consecutive frame buffers to avoid flickering
let faceAbsentFrames = 0;
let gazeAwayFrames = 0;
let phonePresentFrames = 0;
let cleanFocusedFrames = 0;

const STREAK_ABSENT_TRIGGER = 120; // ~8.0s grace period (at ~15fps processing) to prevent alarms during head rotations
const STREAK_GAZE_TRIGGER = 15;   // ~1.0s
const STREAK_PHONE_TRIGGER = 5;    // ~0.3s
const STREAK_FOCUS_TRIGGER = 10;   // ~0.6s

// Timer & session state
let timerInterval = null;
let timeRemaining = 25 * 60; // 25 minutes default
let sessionDuration = 25 * 60;
let isTimerRunning = false;
let isDistracted = false; // Severe distraction (red overlay, timer paused)
let isGazeDistracted = false; // Gaze distraction (no overlay, timer running, logs loss time)
let requireBookInFrame = false; // Default to false (allows looking down at desk as focused by default)
let isTrackingEnabled = false; // Master toggle for AI camera tracking
let isScreenScanEnabled = true; // Master toggle for tab/window monitoring (Screen Scan)
let tabAwayTimestamp = null; // Track when user leaves the tab/window
let updateDial = null; // Sync function for circular clock picker

// Statistics
let totalFocusedSeconds = 0;
let totalSessionSeconds = 0;
let totalLossSeconds = 0;
let distractionCount = 0;

// Tasks
let tasks = [];
let activeTaskId = null;
let isStartTimeManuallyEdited = false;

// Performance optimization (Frame skipping)
let frameCount = 0;
let lastFaceResult = null;
let lastObjectResult = null;
let lastHandResult = null;

// Audio alerts
const alertSound = document.getElementById("sound-alert");

// ==========================================
// DOM ELEMENTS
// ==========================================
const systemStatusText = document.getElementById("system-status-text");
const statusDot = document.querySelector(".status-dot");
const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output-canvas");
const canvasCtx = canvasElement.getContext("2d");
const gazeDot = document.getElementById("gaze-dot");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const cameraPlaceholderText = document.getElementById("camera-placeholder-text");
const btnRetryCamera = document.getElementById("btn-retry-camera");

// Buttons
const btnStart = document.getElementById("btn-start");
const btnPause = document.getElementById("btn-pause");
const btnReset = document.getElementById("btn-reset");
const btnRestart = document.getElementById("btn-restart");
const btnToggleTracker = document.getElementById("btn-toggle-tracker");
const btnCalibrate = document.getElementById("btn-calibrate");
const presetButtons = document.querySelectorAll(".preset-btn[data-minutes]");
const btnTriggerCustom = document.getElementById("custom-time-trigger");
const customMinutesDisplay = document.getElementById("custom-minutes-display");
const customMinutesInput = document.getElementById("custom-minutes");
const timerAmbientGlow = document.getElementById("timer-ambient-glow");

// Stats Displays
const statTotalTime = document.getElementById("stat-total-time");
const statDistractions = document.getElementById("stat-distractions");
const statFocusScore = document.getElementById("stat-focus-score");

// Tasks
const taskForm = document.getElementById("add-task-form");
const taskNameInput = document.getElementById("task-name");
const taskStartTimeInput = document.getElementById("task-start-time");
const taskDurationInput = document.getElementById("task-duration");
const tasksList = document.getElementById("tasks-list");
const activeTaskDisplay = document.getElementById("active-task-display");

// Overlay
const distractionOverlay = document.getElementById("distraction-overlay");
const distractionMessage = document.getElementById("distraction-message");

// Camera Panel Border
const cameraFeedContainer = document.querySelector(".camera-feed-container");

// AI Logs
const logPresence = document.querySelector("#log-presence span");
const logGaze = document.querySelector("#log-gaze span");
const logPhone = document.querySelector("#log-phone span");
const logHands = document.querySelector("#log-hands span");
const logObjects = document.querySelector("#log-objects span");
const toggleBookCheck = document.getElementById("toggle-book-check");
const toggleAiTracking = document.getElementById("toggle-ai-tracking");
const toggleScreenScan = document.getElementById("toggle-screen-scan");

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  setDefaultStartTime();
  loadTasks();
  
  // Sync tracking enablement status with checkbox (handles browser refresh state recovery)
  if (toggleAiTracking) {
    isTrackingEnabled = toggleAiTracking.checked;
  }
  if (toggleScreenScan) {
    isScreenScanEnabled = toggleScreenScan.checked;
  }
  if (isTrackingEnabled) {
    startWebcam(); // Start camera immediately if enabled
    updateTrackerVisibility(false);
  } else {
    stopWebcam();
    updateTrackerVisibility(true);
  }
  
  initAI();      // Load AI models in parallel
  setupTimerPresets();
  setupEventListeners();
  recoverSession(); // Restore session if reload/exit was attempted
});

// Function to update the camera tracker panel visibility and sync the toggle tracker button
function updateTrackerVisibility(hide) {
  const appGrid = document.querySelector(".app-grid");
  const btnSpan = btnToggleTracker.querySelector("span");
  const btnIcon = btnToggleTracker.querySelector("svg");
  
  if (!appGrid || !btnToggleTracker) return;

  if (hide) {
    appGrid.classList.add("hide-tracker-active");
    if (btnSpan) btnSpan.textContent = "Show Tracker";
    btnToggleTracker.title = "Show camera tracker panel";
    btnToggleTracker.classList.add("btn-warning");
    if (btnIcon) btnIcon.innerHTML = `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>`;
  } else {
    appGrid.classList.remove("hide-tracker-active");
    if (btnSpan) btnSpan.textContent = "Hide Tracker";
    btnToggleTracker.title = "Hide camera tracker panel completely";
    btnToggleTracker.classList.remove("btn-warning");
    if (btnIcon) btnIcon.innerHTML = `<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>`;
  }
}

// Get formatted time in Dhaka timezone for ETA display
function getDhakaFormattedETA(date) {
  try {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Dhaka',
      hour: 'numeric',
      minute: 'numeric',
      hour12: true
    });
    return formatter.format(date);
  } catch (err) {
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hr = hours % 12 || 12;
    const min = String(minutes).padStart(2, '0');
    return `${hr}:${min} ${ampm}`;
  }
}

// Sets default task start time to current HH:MM in Dhaka timezone
function setDefaultStartTime() {
  try {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Dhaka',
      hour: 'numeric',
      minute: 'numeric',
      hour12: false
    });
    const parts = formatter.formatToParts(new Date());
    let hr = "00", min = "00";
    parts.forEach(part => {
      if (part.type === 'hour') hr = part.value.padStart(2, "0");
      if (part.type === 'minute') min = part.value.padStart(2, "0");
    });
    taskStartTimeInput.value = `${hr}:${min}`;
  } catch (err) {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    taskStartTimeInput.value = `${hours}:${minutes}`;
  }
}

// Load AI models via CDN files
async function initAI() {
  try {
    systemStatusText.textContent = "Loading WebAssembly runtime...";
    statusDot.className = "status-dot orange";

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );

    systemStatusText.textContent = "Loading Face, Object & Hand Models...";

    const [landmarker, detector, hands] = await Promise.all([
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        },
        runningMode: "VIDEO",
        numFaces: 1
      }),
      ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
        },
        runningMode: "VIDEO",
        scoreThreshold: 0.18
      }),
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 2
      })
    ]);

    faceLandmarker = landmarker;
    objectDetector = detector;
    handLandmarker = hands;

    if (webcamRunning) {
      btnCalibrate.disabled = false;
      systemStatusText.textContent = "Gaze Calibration Required.";
      statusDot.className = "status-dot orange";
    } else {
      systemStatusText.textContent = "AI Ready. Starting camera...";
      statusDot.className = "status-dot orange";
    }
  } catch (error) {
    console.error("AI Initialization Error:", error);
    systemStatusText.textContent = "Failed to load AI models.";
    statusDot.className = "status-dot red";
  }
}

// Stop webcam stream and release camera device access (turning off camera indicator light)
function stopWebcam() {
  if (stream) {
    try {
      stream.getTracks().forEach(track => track.stop());
    } catch (e) {
      console.warn("Failed to stop stream tracks:", e);
    }
    stream = null;
  }
  webcamElement.srcObject = null;
  webcamRunning = false;

  cameraPlaceholder.classList.remove("hidden");
  const placeholderText = cameraPlaceholder.querySelector("p");
  if (placeholderText) {
    placeholderText.textContent = "Camera is turned off (AI Tracking disabled)";
  }
  webcamElement.classList.add("hidden");

  // Clear canvas overlay drawing
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (gazeDot) {
    gazeDot.classList.add("hidden");
  }

  // Set visual status labels to Disabled
  if (logPresence) { logPresence.textContent = "Disabled"; logPresence.className = "status-warn"; }
  if (logGaze) { logGaze.textContent = "Disabled"; logGaze.className = "status-warn"; }
  if (logPhone) { logPhone.textContent = "Disabled"; logPhone.className = ""; }
  if (logHands) { logHands.textContent = "Disabled"; logHands.className = ""; }
  if (logObjects) { logObjects.textContent = "Disabled"; logObjects.className = ""; }
}

// Start webcam stream
async function startWebcam() {
  if (btnRetryCamera) btnRetryCamera.classList.add("hidden");
  if (cameraPlaceholderText) cameraPlaceholderText.textContent = "Requesting camera access...";

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error("Camera API not supported or blocked (insecure context)");
    systemStatusText.textContent = "Camera not supported (Insecure context)";
    statusDot.className = "status-dot red";
    if (cameraPlaceholderText) {
      cameraPlaceholderText.textContent = "⚠️ Camera access is not supported. Ensure you are using HTTPS.";
    }
    return;
  }

  try {
    try {
      // First try with preferred constraints
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (constraintError) {
      console.warn("Preferred constraints failed, trying basic video fallback...", constraintError);
      // Fallback to absolute basic video stream constraints
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
    }

    webcamElement.srcObject = stream;
    
    // Explicitly call play() which is required for mobile browsers
    webcamElement.play().catch(err => {
      console.warn("Failed to play webcam video element:", err);
    });
    
    webcamElement.onloadeddata = () => {
      cameraPlaceholder.classList.add("hidden");
      webcamElement.classList.remove("hidden");
      webcamRunning = true;
      
      // Setup canvas sizes
      canvasElement.width = webcamElement.videoWidth;
      canvasElement.height = webcamElement.videoHeight;
      
      if (faceLandmarker && objectDetector && handLandmarker) {
        btnCalibrate.disabled = false;
        if (isCalibrated) {
          systemStatusText.textContent = "AI Calibrated. Focus system active.";
          statusDot.className = "status-dot green";
        } else {
          systemStatusText.textContent = "Gaze Calibration Required.";
          statusDot.className = "status-dot orange";
        }
      } else {
        systemStatusText.textContent = "Loading AI Models...";
        statusDot.className = "status-dot orange";
      }
      
      // Start processing loop
      requestAnimationFrame(predictLoop);
    };
  } catch (err) {
    console.error("Webcam access failed:", err);
    statusDot.className = "status-dot red";
    
    let userMsg = `Camera Error: ${err.name}`;
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      userMsg = "Camera access was denied. Please click the button below to retry and grant permission.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      userMsg = "No webcam detected. Please plug in or enable your webcam and retry.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      userMsg = "Camera is already in use by another app or tab. Close other apps and retry.";
    }
    
    systemStatusText.textContent = `Error: ${err.name}`;
    if (cameraPlaceholderText) {
      cameraPlaceholderText.textContent = `⚠️ ${userMsg}`;
    }
    if (btnRetryCamera) {
      btnRetryCamera.classList.remove("hidden");
    }
  }
}

// ==========================================
// CORE PROCESSING LOOP (AI DETECT)
// ==========================================
function predictLoop() {
  if (!webcamRunning) return;

  // Stagger execution to ~15 FPS to prevent UI lag on slower systems
  frameCount++;
  if (frameCount % 2 === 0 && webcamElement.currentTime !== lastVideoTime) {
    lastVideoTime = webcamElement.currentTime;
    
    // Run predictions
    runDetections();
  }

  requestAnimationFrame(predictLoop);
}

function runDetections() {
  // Ensure canvas dimensions match webcam video dimensions dynamically (handles device orientation changes)
  if (webcamElement.videoWidth && (canvasElement.width !== webcamElement.videoWidth || canvasElement.height !== webcamElement.videoHeight)) {
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;
  }

  if (!isTrackingEnabled) {
    // Clear canvas and hide overlays
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    logPresence.textContent = "Disabled";
    logPresence.className = "status-warn";
    logGaze.textContent = "Disabled";
    logGaze.className = "status-warn";
    logPhone.textContent = "Disabled";
    logPhone.className = "";
    logHands.textContent = "Disabled";
    logHands.className = "";
    logObjects.textContent = "Disabled";
    logObjects.className = "";
    gazeDot.classList.add("hidden");
    return;
  }

  if (!faceLandmarker || !objectDetector || !handLandmarker) return;

  const timestamp = performance.now();
  
  // Clear canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Stagger model runs to dramatically optimize CPU/GPU load on mobile devices
  // FaceLandmarker: Run every 2nd detection frame
  if (frameCount % 2 === 0 || !lastFaceResult) {
    try {
      lastFaceResult = faceLandmarker.detectForVideo(webcamElement, timestamp);
    } catch (e) {
      console.warn("FaceLandmarker detection failed:", e);
    }
  }
  const faceResult = lastFaceResult || { faceLandmarks: [] };
  
  // ObjectDetector: Run every 6th detection frame (phone/book detections don't need real-time 15fps speed)
  if (frameCount % 6 === 0 || !lastObjectResult) {
    try {
      lastObjectResult = objectDetector.detectForVideo(webcamElement, timestamp);
    } catch (e) {
      console.warn("ObjectDetector detection failed:", e);
    }
  }
  const objectResult = lastObjectResult || { detections: [] };

  // HandLandmarker: Run every 4th detection frame (hand tracking skeleton outlines)
  if (frameCount % 4 === 0 || !lastHandResult) {
    try {
      lastHandResult = handLandmarker.detectForVideo(webcamElement, timestamp);
    } catch (e) {
      console.warn("HandLandmarker detection failed:", e);
    }
  }
  const handResult = lastHandResult || { landmarks: [], worldLandmarks: [], handedness: [] };

  let facePresent = faceResult.faceLandmarks.length > 0;
  let personDetected = false;
  let phoneDetected = false;
  let bookDetected = false;
  let gazeStatus = "neutral"; // neutral, center, book, away, down_no_book
  let detectedLabels = [];
  
  // Scan object detections for distractions and study objects
  if (objectResult.detections) {
    for (const detection of objectResult.detections) {
      const category = detection.categories[0];
      if (category) {
        const name = category.categoryName;
        const score = category.score;
        
        if (name === "person" && score > 0.30) {
          personDetected = true;
        }
        
        // Strict phone check (increased threshold to 0.45 to avoid false positives like headphones or ears)
        if ((name === "cell phone" && score > 0.45) || (name === "remote" && score > 0.45)) {
          phoneDetected = true;
          const displayLabel = name === "cell phone" ? "Cell Phone" : "Handheld Device/Phone";
          const scorePercent = Math.round(score * 100);
          drawObjectBox(detection.boundingBox, `${displayLabel} (${scorePercent}%)`, "red");
          detectedLabels.push(`${displayLabel} (${scorePercent}%)`);
        } else if (name === "book" && score > 0.35) {
          bookDetected = true;
          // Book detection is registered silently for "Strict Book Check", but no visual box is drawn
        }
      }
    }
  }

  // Draw hand skeletons to show hands are always monitored
  let handsCount = 0;
  if (handResult.landmarks && handResult.landmarks.length > 0) {
    handsCount = handResult.landmarks.length;
    handResult.landmarks.forEach(hand => {
      drawHandOutline(hand);
    });
  }

  // Draw face wireframe outline (subtle) for visual feedback
  if (facePresent) {
    drawFaceOutline(faceResult.faceLandmarks[0]);
    
    // Compute face normal vector
    const landmarks = faceResult.faceLandmarks[0];
    const normal = computeFaceNormal(landmarks);
    
    // Handle calibration
    if (isCalibrating) {
      handleCalibrationStep(normal);
    } else if (isCalibrated) {
      gazeStatus = evaluateGaze(normal, bookDetected);
    }
  } else {
    gazeDot.classList.add("hidden");
    
    // Fallback: If face landmarks are lost but person is still detected, they are looking down at desk
    if (personDetected && isCalibrated) {
      if (!requireBookInFrame || bookDetected) {
        gazeStatus = "book";
      } else {
        gazeStatus = "down_no_book";
      }
    }
  }

  // Update AI status logs
  updateLogs(facePresent, personDetected, gazeStatus, phoneDetected, detectedLabels, handsCount);

  // Evaluate distraction rules
  evaluateDistractions(facePresent, personDetected, gazeStatus, phoneDetected);
}

// ==========================================
// 3D GEOMETRIC GAZE MATH
// ==========================================
function computeFaceNormal(landmarks) {
  // Key points: Left eye (33), Right eye (263), Chin (152)
  const A = landmarks[33];
  const B = landmarks[263];
  const C = landmarks[152];

  // Vectors from left eye to right eye, and left eye to chin
  const u = {
    x: B.x - A.x,
    y: B.y - A.y,
    z: B.z - A.z
  };

  const v = {
    x: C.x - A.x,
    y: C.y - A.y,
    z: C.z - A.z
  };

  // Cross Product: u x v (gives the surface normal vector of the face)
  const nx = u.y * v.z - u.z * v.y;
  const ny = u.z * v.x - u.x * v.z;
  const nz = u.x * v.y - u.y * v.x;

  // Normalize to unit vector
  const length = Math.sqrt(nx*nx + ny*ny + nz*nz);
  
  return {
    x: nx / length,
    y: ny / length,
    z: nz / length
  };
}

function handleCalibrationStep(normal) {
  const elapsed = performance.now() - calibrationStartTimestamp;
  
  if (elapsed < CALIBRATION_DURATION_MS) {
    calibrationFrames.push({ x: normal.x, y: normal.y });
    
    // Update progress bar UI
    const percent = Math.min(100, (elapsed / CALIBRATION_DURATION_MS) * 100);
    document.getElementById("calibration-progress-fill").style.width = `${percent}%`;
  } else {
    // End calibration
    isCalibrating = false;
    isCalibrated = true;
    
    // Average baseline coordinates
    const sum = calibrationFrames.reduce((acc, f) => ({ x: acc.x + f.x, y: acc.y + f.y }), { x: 0, y: 0 });
    baselineGaze = {
      x: sum.x / calibrationFrames.length,
      y: sum.y / calibrationFrames.length
    };
    
    document.getElementById("calibration-progress-bar").classList.add("hidden");
    btnCalibrate.textContent = "Recalibrate Gaze";
    btnCalibrate.className = "btn btn-outline";
    
    systemStatusText.textContent = "AI Calibrated. Focus system active.";
    statusDot.className = "status-dot green";
    
    // Enable timer settings and buttons
    btnStart.disabled = false;
  }
}

function evaluateGaze(normal, bookDetected) {
  const diffX = normal.x - baselineGaze.x;
  const diffY = -(normal.y - baselineGaze.y); // Invert sign so looking down is positive, looking up is negative

  // Draw Gaze pointer dot on UI (passing bookDetected status)
  drawGazeDot(diffX, diffY, bookDetected);

  // Pitch evaluation for looking down
  if (diffY > PITCH_THRESHOLD) {
    // looking down (no maximum down limit as long as face is detected)
    if (!requireBookInFrame || bookDetected) {
      return "book"; // focused on book (or book checking is disabled)!
    } else {
      return "down_no_book"; // looking down but no book on desk
    }
  }

  return "center"; // default to focused (looking straight/left/right/up)
}

// ==========================================
// DISTRACTION & FOCUS MANAGEMENT
// ==========================================
function evaluateDistractions(facePresent, personDetected, gazeStatus, phoneDetected) {
  const userPresent = facePresent || personDetected;
  
  // Gaze is away if status is "away" or "down_no_book" (no book when looking down)
  const isGazeAway = isCalibrated && (gazeStatus === "away" || gazeStatus === "down_no_book");
  
  // Update consecutive frame streaks
  if (!userPresent) {
    faceAbsentFrames++;
  } else {
    faceAbsentFrames = 0;
  }

  if (isGazeAway) {
    gazeAwayFrames++;
  } else {
    gazeAwayFrames = 0;
  }

  if (phoneDetected) {
    phonePresentFrames++;
  } else {
    phonePresentFrames = 0;
  }

  // Handle Mild Gaze Distraction (Timer does NOT stop, no red overlay screen)
  if (gazeAwayFrames >= STREAK_GAZE_TRIGGER) {
    if (!isGazeDistracted) {
      isGazeDistracted = true;
      cameraFeedContainer.className = "camera-feed-container distracted"; // orange/red visual borders
      if (isTimerRunning) {
        distractionCount++;
        if (activeTaskId) {
          const activeTask = tasks.find(t => t.id === activeTaskId);
          if (activeTask) {
            if (!activeTask.distractionCount) activeTask.distractionCount = 0;
            activeTask.distractionCount++;
            saveTasks();
            const statsEl = document.getElementById(`task-stats-${activeTask.id}`);
            if (statsEl) {
              statsEl.textContent = `🎯 Focused: ${formatSeconds(activeTask.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(activeTask.lossSeconds || 0)} | ⚠️ Distractions: ${activeTask.distractionCount || 0}`;
            }
          }
        }
        updateStatsDisplay();
      }
    }
  } else if (gazeAwayFrames === 0 && !isDistracted) {
    if (isGazeDistracted) {
      isGazeDistracted = false;
      cameraFeedContainer.className = "camera-feed-container focusing";
    }
  }

  // Handle Severe Distraction (Phone or Absent - timer stops, red screen shows)
  const isSevereDistracted = (faceAbsentFrames >= STREAK_ABSENT_TRIGGER) || (phonePresentFrames >= STREAK_PHONE_TRIGGER);

  if (isSevereDistracted) {
    cleanFocusedFrames = 0;
    if (!isDistracted) {
      triggerDistraction(faceAbsentFrames, phonePresentFrames);
    }
  } else {
    // Restore focus if they are present, looking straight/down, no phone, and browser is in Fullscreen Mode
    const isFullscreen = document.fullscreenElement !== null;
    if (userPresent && !isGazeAway && !phoneDetected && isFullscreen) {
      cleanFocusedFrames++;
      if (cleanFocusedFrames >= STREAK_FOCUS_TRIGGER && isDistracted) {
        triggerFocusRestore();
      }
    } else {
      cleanFocusedFrames = 0;
    }
  }
}

function triggerDistraction(absent, phone, type = null) {
  isDistracted = true;
  distractionCount++;
  if (activeTaskId) {
    const activeTask = tasks.find(t => t.id === activeTaskId);
    if (activeTask) {
      if (!activeTask.distractionCount) activeTask.distractionCount = 0;
      activeTask.distractionCount++;
      saveTasks();
      const statsEl = document.getElementById(`task-stats-${activeTask.id}`);
      if (statsEl) {
        statsEl.textContent = `🎯 Focused: ${formatSeconds(activeTask.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(activeTask.lossSeconds || 0)} | ⚠️ Distractions: ${activeTask.distractionCount || 0}`;
      }
    }
  }
  updateStatsDisplay();

  // Record timestamp when user navigates away from tab/window
  if ((type === "blur" || type === "visibility" || type === "fullscreen") && !tabAwayTimestamp) {
    tabAwayTimestamp = Date.now();
    // Pause interval during absence to prevent double counting if throttled ticks occur
    pauseTimerInterval();
  }

  // Dynamic alert warning message
  let msg = "Please focus back on your study to resume the timer.";
  if (type === "blur") {
    msg = "⚠️ <strong>LOCKDOWN VIOLATION!</strong> Window lost focus! You opened another application or window. Keyboard/Mouse is disabled. Return and stay focused on your study!";
  } else if (type === "visibility") {
    msg = "⚠️ <strong>LOCKDOWN VIOLATION!</strong> Tab switched! Swapping tabs is strictly prohibited during focus sessions. Keyboard/Mouse is disabled.";
  } else if (type === "fullscreen") {
    msg = "⚠️ <strong>ABSOLUTE LOCKDOWN!</strong> Fullscreen exited. You CANNOT disable Fullscreen mode under any circumstances! Keyboard/Mouse is fully locked. Click anywhere on the screen to force re-enter Fullscreen mode.";
  } else if (type === "reload") {
    msg = "⚠️ <strong>LOCKDOWN ESCAPE ATTEMPT DETECTED!</strong> Page reload detected. The timer has resumed and keyboard/mouse is locked. You must stay focused in Fullscreen mode!";
  } else if (phone >= STREAK_PHONE_TRIGGER) {
    msg = "⚠️ Phone detected! Please put your phone away to continue.";
  } else if (absent >= STREAK_ABSENT_TRIGGER) {
    msg = "⚠️ No face detected. Please return to your desk.";
  }

  // If not in Fullscreen, prompt to re-enter
  if (!document.fullscreenElement) {
    msg += `<br><button id="btn-re-fullscreen" class="btn btn-primary" style="margin-top: 1.25rem; padding: 0.6rem 1.2rem; font-size: 0.85rem; width: auto; display: inline-flex;">Re-enter Fullscreen</button>`;
  }

  distractionMessage.innerHTML = msg;
  distractionOverlay.classList.remove("hidden");
  setTimeout(() => distractionOverlay.classList.add("show"), 10);
  
  // Attach event listener for the re-fullscreen button
  const reFsBtn = document.getElementById("btn-re-fullscreen");
  if (reFsBtn) {
    reFsBtn.onclick = (e) => {
      e.stopPropagation();
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen()
          .then(() => triggerFocusRestore())
          .catch(err => {
            console.warn("Fullscreen request rejected, restoring focus anyway:", err);
            triggerFocusRestore();
          });
      } else {
        triggerFocusRestore(); // Fallback for iPhone Safari
      }
    };
  }
  
  // Style changes to camera feed border
  cameraFeedContainer.className = "camera-feed-container distracted";
  document.body.classList.add("distracted-active");

  // Change progress circle and timer clock to red glow when distracted
  const progressCircle = document.getElementById("progress-circle");
  if (progressCircle) {
    progressCircle.style.stroke = "url(#timer-distracted-gradient)";
    progressCircle.classList.add("distracted");
  }
  const timerClock = document.getElementById("timer-clock");
  if (timerClock) {
    timerClock.classList.add("distracted-glow");
  }

  // Play alert sound (looping until they focus back)
  alertSound.loop = true;
  alertSound.currentTime = 0;
  alertSound.play().catch(e => console.log("Sound alert blocked by browser play policy."));

  // Do NOT pause the timer interval anymore! The timer keeps ticking down as punishment (Loss Time).
  if (isTimerRunning) {
    systemStatusText.textContent = "Lockdown active - Wasting study time!";
    statusDot.className = "status-dot red";
  }
}

function triggerFocusRestore() {
  isDistracted = false;
  isGazeDistracted = false;
  
  // Calculate and apply loss time spent on other windows/tabs
  if (tabAwayTimestamp) {
    const elapsedSecondsAway = Math.round((Date.now() - tabAwayTimestamp) / 1000);
    const sessionLoss = Math.min(timeRemaining, elapsedSecondsAway);
    totalLossSeconds += sessionLoss;
    totalSessionSeconds += sessionLoss;
    
    // Increment active task stats
    if (activeTaskId) {
      const activeTask = tasks.find(t => t.id === activeTaskId);
      if (activeTask) {
        if (!activeTask.lossSeconds) activeTask.lossSeconds = 0;
        if (!activeTask.sessionSeconds) activeTask.sessionSeconds = 0;
        activeTask.lossSeconds += sessionLoss;
        activeTask.sessionSeconds += sessionLoss;
        saveTasks();
        const statsEl = document.getElementById(`task-stats-${activeTask.id}`);
        if (statsEl) {
          statsEl.textContent = `🎯 Focused: ${formatSeconds(activeTask.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(activeTask.lossSeconds || 0)} | ⚠️ Distractions: ${activeTask.distractionCount || 0}`;
        }
      }
    }
    
    timeRemaining = Math.max(0, timeRemaining - sessionLoss);
    tabAwayTimestamp = null;
    
    updateTimerDisplay();
    updateStatsDisplay();
    
    if (timeRemaining <= 0) {
      completeSession();
      return;
    }
  }
  
  // Stop looping alert sound
  alertSound.loop = false;
  alertSound.pause();
  alertSound.currentTime = 0;
  
  // Hide overlay
  distractionOverlay.classList.remove("show");
  setTimeout(() => {
    if (!isDistracted) distractionOverlay.classList.add("hidden");
  }, 400);

  cameraFeedContainer.className = "camera-feed-container focusing";
  document.body.classList.remove("distracted-active");

  // Restore progress circle and timer clock to green glow when focusing
  const progressCircle = document.getElementById("progress-circle");
  if (progressCircle) {
    progressCircle.style.stroke = "url(#timer-gradient)";
    progressCircle.classList.remove("distracted");
  }
  const timerClock = document.getElementById("timer-clock");
  if (timerClock) {
    timerClock.classList.remove("distracted-glow");
  }
  
  // Resume Timer if user had session active
  if (isTimerRunning) {
    startTimerInterval();
    systemStatusText.textContent = "Focus session active (Lockdown Mode).";
    statusDot.className = "status-dot green";
  } else {
    systemStatusText.textContent = "Focus system ready.";
    statusDot.className = "status-dot green";
  }
}

// ==========================================
// TIMER SYSTEM
// ==========================================
function setupTimerPresets() {
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      if (isTimerRunning) return;
      presetButtons.forEach(b => b.classList.remove("active"));
      if (btnTriggerCustom) btnTriggerCustom.classList.remove("active");
      if (customMinutesDisplay) customMinutesDisplay.value = "";
      btn.classList.add("active");
      
      const minutes = parseInt(btn.dataset.minutes);
      setTimerDuration(minutes);
      if (updateDial) updateDial(minutes);
    });
  });

  initDialPicker();
}

function initDialPicker() {
  const dialModal = document.getElementById("dial-modal");
  const btnCloseDial = document.getElementById("btn-close-dial");
  const btnConfirmCustomTime = document.getElementById("btn-confirm-custom-time");

  const dialSvg = document.querySelector(".dial-svg");
  const dialKnob = document.getElementById("dial-knob");
  const dialActiveTrack = document.getElementById("dial-active-track");
  const dialMinutesValue = document.getElementById("dial-minutes-value");
  const customMinutesInput = document.getElementById("custom-minutes");

  if (!dialSvg || !dialModal) return;

  const radius = 65;
  const centerX = 80;
  const centerY = 80;
  const circumference = 2 * Math.PI * radius;

  let isDragging = false;
  let tempMinutes = 25;

  // Set the stroke dasharray for active track
  if (dialActiveTrack) {
    dialActiveTrack.style.strokeDasharray = `${circumference} ${circumference}`;
  }

  // Open modal
  if (btnTriggerCustom) {
    btnTriggerCustom.addEventListener("click", () => {
      if (isTimerRunning) return;
      dialModal.classList.remove("hidden");
      setTimeout(() => dialModal.classList.add("show"), 10);
      
      // Set dial to current session duration
      tempMinutes = Math.round(sessionDuration / 60);
      updateDialVisuals(tempMinutes);
    });
  }

  // Close modal functions
  function closeModal() {
    dialModal.classList.remove("show");
    setTimeout(() => dialModal.classList.add("hidden"), 300);
  }

  if (btnCloseDial) {
    btnCloseDial.addEventListener("click", closeModal);
  }
  
  dialModal.addEventListener("click", (e) => {
    if (e.target === dialModal) closeModal();
  });

  // Confirm selection
  if (btnConfirmCustomTime) {
    btnConfirmCustomTime.addEventListener("click", () => {
      presetButtons.forEach(b => b.classList.remove("active"));
      if (btnTriggerCustom) btnTriggerCustom.classList.add("active");
      setTimerDuration(tempMinutes);
      if (customMinutesDisplay) customMinutesDisplay.value = tempMinutes;
      closeModal();
    });
  }

  function updateDialVisuals(minutes) {
    if (!dialKnob || !dialActiveTrack || !dialMinutesValue) return;

    // Clamp display minutes between 1 and 60 for visual rotation
    const displayMin = Math.max(1, Math.min(60, minutes));
    const angle = (displayMin / 60) * 2 * Math.PI;

    // Calculate knob position
    const knobX = centerX + radius * Math.sin(angle);
    const knobY = centerY - radius * Math.cos(angle);

    dialKnob.setAttribute("cx", knobX);
    dialKnob.setAttribute("cy", knobY);

    // Update active fill track
    const offset = circumference - (displayMin / 60) * circumference;
    dialActiveTrack.style.strokeDashoffset = offset;

    // Update text and input values
    dialMinutesValue.textContent = minutes;
    if (customMinutesInput) customMinutesInput.value = minutes;
  }

  updateDial = (minutes) => {
    tempMinutes = minutes;
    updateDialVisuals(minutes);
  };

  function handleMove(clientX, clientY) {
    const rect = dialSvg.getBoundingClientRect();
    const x = clientX - rect.left - centerX;
    const y = clientY - rect.top - centerY;

    let angle = Math.atan2(y, x) + Math.PI / 2;
    if (angle < 0) angle += 2 * Math.PI;

    let minutes = Math.round((angle / (2 * Math.PI)) * 60);
    if (minutes === 0) minutes = 60;

    tempMinutes = minutes;
    updateDialVisuals(minutes);
  }

  // Mouse events
  dialSvg.addEventListener("mousedown", (e) => {
    isDragging = true;
    handleMove(e.clientX, e.clientY);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    handleMove(e.clientX, e.clientY);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Touch events for mobile
  dialSvg.addEventListener("touchstart", (e) => {
    isDragging = true;
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX, e.touches[0].clientY);
    }
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX, e.touches[0].clientY);
    }
  });

  window.addEventListener("touchend", () => {
    isDragging = false;
  });

  // Listen to input changes in modal to sync dial
  if (customMinutesInput) {
    customMinutesInput.addEventListener("input", () => {
      const val = parseInt(customMinutesInput.value);
      if (val > 0) {
        tempMinutes = val;
        updateDialVisuals(val);
      }
    });
  }

  // Init position to 25 mins
  updateDialVisuals(25);
}

function setTimerDuration(minutes) {
  sessionDuration = minutes * 60;
  timeRemaining = sessionDuration;
  updateTimerDisplay();
}

function updateTimerDisplay() {
  const m = Math.floor(timeRemaining / 60);
  const s = timeRemaining % 60;
  document.getElementById("timer-clock").textContent = 
    `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    
  // Circle stroke offset calculation
  const circle = document.getElementById("progress-circle");
  const radius = circle.r.baseVal.value;
  const circumference = radius * 2 * Math.PI;
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  
  const offset = circumference - (timeRemaining / sessionDuration) * circumference;
  circle.style.strokeDashoffset = offset;
}

function startTimer() {
  if (isTrackingEnabled && !isCalibrated) {
    alert("Please calibrate the gaze tracker before starting the focus session!");
    return;
  }

  // Always request Fullscreen Mode to strictly block switching tabs or opening other apps
  if (document.documentElement.requestFullscreen) {
    document.documentElement.requestFullscreen().catch(err => {
      console.warn("Fullscreen request rejected:", err);
    });
  }
  
  isTimerRunning = true;
  btnStart.classList.add("hidden");
  
  // Show ETA (Estimated finishing time)
  const timerEta = document.getElementById("timer-eta");
  if (timerEta) {
    const finishTime = new Date(Date.now() + timeRemaining * 1000);
    timerEta.textContent = `Ends at: ${getDhakaFormattedETA(finishTime)}`;
    timerEta.classList.remove("hidden");
  }
  
  // Always hide Pause/Reset/Restart control buttons to enforce lockdown
  btnPause.classList.add("hidden");
  btnReset.classList.add("hidden");
  btnRestart.classList.add("hidden");
  document.body.classList.add("lockdown-active");
  document.body.classList.add("timer-running");
  
  // Disable presets inputs while timer runs
  presetButtons.forEach(btn => btn.disabled = true);
  customMinutesInput.disabled = true;

  // Save session state to localStorage to prevent escape via reload
  localStorage.setItem("study_session_running", "true");
  localStorage.setItem("study_session_end_time", Date.now() + timeRemaining * 1000);
  localStorage.setItem("study_session_duration", sessionDuration);
  localStorage.setItem("study_session_total_focused", totalFocusedSeconds);
  localStorage.setItem("study_session_total_loss", totalLossSeconds);
  localStorage.setItem("study_session_total_session", totalSessionSeconds);
  localStorage.setItem("study_session_distractions", distractionCount);

  if (!isDistracted) {
    startTimerInterval();
    systemStatusText.textContent = "Focus session active (Lockdown Mode).";
    statusDot.className = "status-dot green";
  }
}

function startTimerInterval() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    totalSessionSeconds++;
    
    // Find active task
    let activeTask = null;
    if (activeTaskId) {
      activeTask = tasks.find(t => t.id === activeTaskId);
    }
    if (activeTask) {
      if (!activeTask.sessionSeconds) activeTask.sessionSeconds = 0;
      activeTask.sessionSeconds++;
    }
    
    if (isTimerRunning) {
      // If distracted or gaze distracted, count as loss time (timer does not pause to prevent escaping)
      if (isDistracted || isGazeDistracted) {
        totalLossSeconds++;
        if (activeTask) {
          if (!activeTask.lossSeconds) activeTask.lossSeconds = 0;
          activeTask.lossSeconds++;
        }
        timeRemaining--;
      } else {
        totalFocusedSeconds++;
        if (activeTask) {
          if (!activeTask.focusedSeconds) activeTask.focusedSeconds = 0;
          activeTask.focusedSeconds++;
        }
        timeRemaining--;
      }
      
      updateTimerDisplay();
      
      // Keep localStorage state updated
      localStorage.setItem("study_session_end_time", Date.now() + timeRemaining * 1000);
      localStorage.setItem("study_session_total_focused", totalFocusedSeconds);
      localStorage.setItem("study_session_total_loss", totalLossSeconds);
      localStorage.setItem("study_session_total_session", totalSessionSeconds);
      localStorage.setItem("study_session_distractions", distractionCount);

      // Save tasks and update UI if active task statistics updated
      if (activeTask) {
        saveTasks();
        const statsEl = document.getElementById(`task-stats-${activeTask.id}`);
        if (statsEl) {
          statsEl.textContent = `🎯 Focused: ${formatSeconds(activeTask.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(activeTask.lossSeconds || 0)} | ⚠️ Distractions: ${activeTask.distractionCount || 0}`;
        }
      }

      // Session finished
      if (timeRemaining <= 0) {
        completeSession();
      }
    }
    
    updateStatsDisplay();
  }, 1000);
}

function pauseTimer() {
  document.body.classList.remove("lockdown-active");
  document.body.classList.remove("timer-running");
  isTimerRunning = false;
  btnPause.classList.add("hidden");
  btnStart.classList.remove("hidden");
  
  tabAwayTimestamp = null;
  
  // Hide ETA
  const timerEta = document.getElementById("timer-eta");
  if (timerEta) {
    timerEta.classList.add("hidden");
  }
  
  pauseTimerInterval();
  
  // Stop alert sound
  alertSound.loop = false;
  alertSound.pause();
  alertSound.currentTime = 0;
  
  localStorage.removeItem("study_session_running");
  
  systemStatusText.textContent = "Session paused.";
  statusDot.className = "status-dot orange";
  
  // Hide distraction overlay and restore normal feed classes on pause
  if (isDistracted || isGazeDistracted) {
    triggerFocusRestore();
  }
}

function pauseTimerInterval() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function resetTimer() {
  pauseTimer();
  timeRemaining = sessionDuration;
  updateTimerDisplay();
  
  // Restore control buttons visibility
  btnStart.classList.remove("hidden");
  btnReset.classList.remove("hidden");
  btnRestart.classList.remove("hidden");
  btnPause.classList.add("hidden");
  
  // Enable presets inputs
  presetButtons.forEach(btn => btn.disabled = false);
  customMinutesInput.disabled = false;
  
  localStorage.removeItem("study_session_running");
}

function restartSession() {
  if (confirm("Are you sure you want to restart the focus session and clear all stats?")) {
    pauseTimer();
    
    // Reset stats
    totalFocusedSeconds = 0;
    totalSessionSeconds = 0;
    totalLossSeconds = 0;
    distractionCount = 0;

    // Reset active task stats if selected
    if (activeTaskId) {
      const activeTask = tasks.find(t => t.id === activeTaskId);
      if (activeTask) {
        activeTask.focusedSeconds = 0;
        activeTask.lossSeconds = 0;
        activeTask.sessionSeconds = 0;
        activeTask.distractionCount = 0;
        saveTasks();
        renderTasks();
      }
    }
    
    // Clear statistics from localStorage
    localStorage.removeItem("study_session_running");
    localStorage.removeItem("study_session_end_time");
    localStorage.removeItem("study_session_duration");
    localStorage.removeItem("study_session_total_focused");
    localStorage.removeItem("study_session_total_loss");
    localStorage.removeItem("study_session_total_session");
    localStorage.removeItem("study_session_distractions");
    
    // Reset timer to preset duration
    timeRemaining = sessionDuration;
    updateTimerDisplay();
    
    // Update stats displays
    updateStatsDisplay();
    
    // Restore control buttons visibility
    btnStart.classList.remove("hidden");
    btnReset.classList.remove("hidden");
    btnRestart.classList.remove("hidden");
    btnPause.classList.add("hidden");
    
    // Enable presets and custom inputs
    presetButtons.forEach(btn => btn.disabled = false);
    customMinutesInput.disabled = false;
    
    systemStatusText.textContent = "Session restarted and stats cleared.";
    statusDot.className = "status-dot orange";
  }
}

function completeSession() {
  pauseTimer();
  alertSound.play();
  alert("🎉 Congratulations! You completed your focus study session!");
  resetTimer();
  
  localStorage.removeItem("study_session_running");
}

function formatSeconds(seconds) {
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s}s`;
  }
  return `${seconds}s`;
}

function updateStatsDisplay() {
  let focused = totalFocusedSeconds;
  let loss = totalLossSeconds;
  let total = totalSessionSeconds;
  let distractions = distractionCount;

  if (activeTaskId) {
    const activeTask = tasks.find(t => t.id === activeTaskId);
    if (activeTask) {
      focused = activeTask.focusedSeconds || 0;
      loss = activeTask.lossSeconds || 0;
      total = activeTask.sessionSeconds || 0;
      distractions = activeTask.distractionCount || 0;
    }
  }

  // Format total focused time
  statTotalTime.textContent = formatSeconds(focused);

  // Format total loss time
  const lossTimeElement = document.getElementById("stat-loss-time");
  if (lossTimeElement) {
    lossTimeElement.textContent = formatSeconds(loss);
  }

  // Format distraction count
  if (statDistractions) {
    statDistractions.textContent = distractions;
  }

  // Calculate focus score
  if (total > 0) {
    const score = Math.round((focused / total) * 100);
    statFocusScore.textContent = `${score}%`;
  } else {
    statFocusScore.textContent = "100%";
  }
}

// ==========================================
// TASK MANAGER (SIDEBAR)
// ==========================================
function setupEventListeners() {
  // Timer Controls
  btnStart.addEventListener("click", startTimer);
  btnPause.addEventListener("click", pauseTimer);
  btnReset.addEventListener("click", resetTimer);
  btnRestart.addEventListener("click", restartSession);

  // Toggle camera tracking panel completely
  btnToggleTracker.addEventListener("click", () => {
    const appGrid = document.querySelector(".app-grid");
    if (appGrid) {
      const isHidden = appGrid.classList.contains("hide-tracker-active");
      updateTrackerVisibility(!isHidden);
    }
  });

  // Calibration
  btnCalibrate.addEventListener("click", () => {
    isCalibrating = true;
    calibrationFrames = [];
    calibrationStartTimestamp = performance.now();
    document.getElementById("calibration-progress-bar").classList.remove("hidden");
    document.getElementById("calibration-progress-fill").style.width = "0%";
    btnCalibrate.textContent = "Calibrating...";
    
    systemStatusText.textContent = "Keep looking straight at the screen...";
    statusDot.className = "status-dot orange";
  });

  // Camera retry
  if (btnRetryCamera) {
    btnRetryCamera.addEventListener("click", () => {
      startWebcam();
    });
  }

  // Task submit
  taskForm.addEventListener("submit", (e) => {
    e.preventDefault();
    addTask();
  });

  // Track if user manually edits task starting time
  taskStartTimeInput.addEventListener("input", () => {
    isStartTimeManuallyEdited = true;
  });
  taskStartTimeInput.addEventListener("change", () => {
    isStartTimeManuallyEdited = true;
  });

  // AI Tracking Master Toggle
  toggleAiTracking.addEventListener("change", () => {
    isTrackingEnabled = toggleAiTracking.checked;
    
    if (!isTrackingEnabled) {
      systemStatusText.textContent = "AI Tracking Disabled.";
      statusDot.className = "status-dot orange";
      
      // Stop camera and release device to turn off green light
      stopWebcam();
      
      // Hide tracker panel
      updateTrackerVisibility(true);
      
      // Stop alert sound if it was playing
      alertSound.loop = false;
      alertSound.pause();
      alertSound.currentTime = 0;
      
      // Hide distraction overlay
      if (isDistracted || isGazeDistracted) {
        triggerFocusRestore();
      }
      
      if (isTimerRunning) {
        systemStatusText.textContent = "Focus session active (Lockdown Mode).";
        statusDot.className = "status-dot green";
      } else {
        systemStatusText.textContent = "AI Tracking Disabled.";
        statusDot.className = "status-dot orange";
      }
    } else {
      // Re-initialize and start webcam
      startWebcam();
      
      // Show tracker panel
      updateTrackerVisibility(false);
      
      if (isTimerRunning) {
        // Request fullscreen to lock down
        if (document.documentElement.requestFullscreen) {
          document.documentElement.requestFullscreen().catch(err => console.warn(err));
        }
        
        if (isCalibrated) {
          systemStatusText.textContent = "Focus session active (Lockdown Mode).";
          statusDot.className = "status-dot green";
        } else {
          systemStatusText.textContent = "Gaze Calibration Required.";
          statusDot.className = "status-dot orange";
        }
      } else {
        if (isCalibrated) {
          systemStatusText.textContent = "AI Active. Focus system active.";
          statusDot.className = "status-dot green";
        } else {
          systemStatusText.textContent = "Gaze Calibration Required.";
          statusDot.className = "status-dot orange";
        }
      }
    }
  });

  // Strict Book Verification Toggle
  toggleBookCheck.addEventListener("change", () => {
    requireBookInFrame = toggleBookCheck.checked;
  });

  // Screen Monitor (Tab/Window Monitoring) Toggle
  if (toggleScreenScan) {
    toggleScreenScan.addEventListener("change", () => {
      isScreenScanEnabled = toggleScreenScan.checked;
    });
  }

  // Toggle Camera Feed Preview (Privacy Screen Toggle)
  const toggleCameraPreview = document.getElementById("toggle-camera-preview");
  const privacyOverlay = document.getElementById("privacy-overlay");
  
  toggleCameraPreview.addEventListener("change", () => {
    if (toggleCameraPreview.checked) {
      cameraFeedContainer.classList.remove("feed-hidden");
      privacyOverlay.classList.add("hidden");
    } else {
      cameraFeedContainer.classList.add("feed-hidden");
      privacyOverlay.classList.remove("hidden");
    }
  });

  // Window Blur, Visibility & Fullscreen Exit Distraction Enforcements
  window.addEventListener("blur", () => {
    if (isScreenScanEnabled && isTimerRunning) {
      if (!isDistracted) {
        triggerDistraction(0, 0, "blur");
      } else if (!tabAwayTimestamp) {
        tabAwayTimestamp = Date.now();
        pauseTimerInterval();
      }
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (isScreenScanEnabled && isTimerRunning) {
      if (document.hidden) {
        if (!isDistracted) {
          triggerDistraction(0, 0, "visibility");
        } else if (!tabAwayTimestamp) {
          tabAwayTimestamp = Date.now();
          pauseTimerInterval();
        }
      }
    }
  });

  document.addEventListener("fullscreenchange", () => {
    const isFullscreen = document.fullscreenElement !== null;
    if (!isFullscreen) {
      if (isScreenScanEnabled && isTimerRunning) {
        if (!isDistracted) {
          triggerDistraction(0, 0, "fullscreen");
        } else if (!tabAwayTimestamp) {
          tabAwayTimestamp = Date.now();
          pauseTimerInterval();
        }
      }
    } else {
      if (isTimerRunning && isDistracted) {
        triggerFocusRestore();
      }
    }
  });

  // Prevent leaving or reloading the tab during active focus session
  window.addEventListener("beforeunload", (e) => {
    if (isTimerRunning) {
      e.preventDefault();
      e.returnValue = "Your study session is active! You cannot reload or exit.";
      return e.returnValue;
    }
  });

  // Block all keyboard inputs during active lockdown study sessions
  window.addEventListener("keydown", (e) => {
    if (isTimerRunning) {
      // Allow keyboard inputs if typing inside the task form inputs or presets inputs
      if (e.target && (e.target.id === "task-name" || e.target.id === "task-duration" || e.target.id === "task-start-time" || e.target.id === "custom-minutes")) {
        return; // Allow typing in text inputs
      }

      e.preventDefault();
      e.stopPropagation();
      
      // If they tried to refresh, show warning alert
      if (e.key === "F5" || ((e.ctrlKey || e.metaKey) && e.key === "r")) {
        alert("⚠️ Keyboard inputs and page reloads are disabled during active study sessions!");
      }
    }
  }, true);

  // Re-request fullscreen on any document click if they somehow escaped fullscreen
  document.addEventListener("click", () => {
    if (isTimerRunning && !document.fullscreenElement) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen()
          .then(() => {
            if (isDistracted) triggerFocusRestore();
          })
          .catch(err => {
            console.warn("Fullscreen request rejected, restoring focus anyway:", err);
            if (isDistracted) triggerFocusRestore();
          });
      } else {
        if (isDistracted) triggerFocusRestore();
      }
    }
  });

  // Handle device orientation changes or window resize to prevent camera stretching/rotation bugs
  let lastOrientation = window.innerHeight > window.innerWidth ? "portrait" : "landscape";
  let resizeTimeout;

  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      const currentOrientation = window.innerHeight > window.innerWidth ? "portrait" : "landscape";
      
      if (currentOrientation !== lastOrientation) {
        lastOrientation = currentOrientation;
        
        if (webcamRunning && isTrackingEnabled) {
          console.log("Device orientation change detected. Restarting camera and resetting calibration...");
          
          // Reset calibration since axis orientation has changed
          isCalibrated = false;
          btnCalibrate.textContent = "Calibrate Gaze";
          btnCalibrate.className = "btn btn-secondary";
          
          stopWebcam();
          startWebcam();
        }
      } else {
        // Just window resize without orientation change (e.g. desktop resize)
        if (webcamRunning && isTrackingEnabled) {
          stopWebcam();
          startWebcam();
        }
      }
    }, 500); // 500ms debounce
  });
}

function addTask() {
  const name = taskNameInput.value.trim();
  const startTime = taskStartTimeInput.value;
  const duration = parseInt(taskDurationInput.value);

  if (!name || !startTime || !duration) return;

  // Calculate target end time
  const [h, m] = startTime.split(":").map(Number);
  const endMin = (m + duration) % 60;
  const endHrs = (h + Math.floor((m + duration) / 60)) % 24;
  const endTime = `${String(endHrs).padStart(2, "0")}:${String(endMin).padStart(2, "0")}`;

  const task = {
    id: Date.now().toString(),
    name,
    startTime,
    endTime,
    duration,
    date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
    completed: false,
    focusedSeconds: 0,
    lossSeconds: 0,
    sessionSeconds: 0,
    distractionCount: 0
  };

  tasks.push(task);
  activeTaskId = task.id;
  activeTaskDisplay.textContent = `Studying: ${task.name}`;
  saveTasks();
  renderTasks();
  updateStatsDisplay();

  // Reset task input fields
  taskNameInput.value = "";
  isStartTimeManuallyEdited = false;
  setDefaultStartTime();
}

function renderTasks() {
  tasksList.innerHTML = "";
  
  if (tasks.length === 0) {
    tasksList.innerHTML = `<li class="task-muted-text" style="color: var(--text-muted); font-size: 0.85rem; text-align: center; margin-top: 1rem;">No tasks added yet.</li>`;
    activeTaskDisplay.textContent = "No active task selected";
    activeTaskId = null;
    return;
  }

  tasks.forEach(task => {
    const li = document.createElement("li");
    li.className = `task-item ${task.completed ? "completed" : ""} ${task.id === activeTaskId ? "active" : ""}`;
    
    // Auto format 24h to 12h clock
    const formattedStart = format12h(task.startTime);
    const formattedEnd = format12h(task.endTime);
    
    // Resolve date safely
    let taskDate = task.date;
    if (!taskDate) {
      const parsedId = parseInt(task.id);
      if (!isNaN(parsedId)) {
        taskDate = new Date(parsedId).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      } else {
        taskDate = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      }
    }

    li.innerHTML = `
      <div class="task-checkbox-container">
        <input type="checkbox" ${task.completed ? "checked" : ""}>
        <div class="task-checkmark">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
      </div>
      <div class="task-details">
        <span class="task-title" title="${task.name}">${task.name}</span>
        <span class="task-meta-sub" style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-top: 0.15rem;">
          📅 ${taskDate} | 🕒 ${formattedStart} - ${formattedEnd} (${task.duration}m)
        </span>
        <span class="task-stats-sub" id="task-stats-${task.id}" style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-top: 0.15rem;">
          🎯 Focused: ${formatSeconds(task.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(task.lossSeconds || 0)} | ⚠️ Distractions: ${task.distractionCount || 0}
        </span>
      </div>
      <button class="task-delete">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
      </button>
    `;

    // Handle check complete
    const checkbox = li.querySelector("input[type='checkbox']");
    checkbox.addEventListener("change", (e) => {
      e.stopPropagation();
      task.completed = checkbox.checked;
      
      // Clear active status if task is completed
      if (task.completed && activeTaskId === task.id) {
        activeTaskId = null;
        activeTaskDisplay.textContent = "No active task selected";
      }
      
      saveTasks();
      renderTasks();
      updateStatsDisplay();
    });

    // Prevent checkbox container clicks from bubbling to li click handler
    const checkboxContainer = li.querySelector(".task-checkbox-container");
    if (checkboxContainer) {
      checkboxContainer.addEventListener("click", (e) => {
        e.stopPropagation();
      });
    }

    // Handle click to set active task
    li.addEventListener("click", (e) => {
      // If the click was on the checkbox or delete button or their children, do not toggle active state
      if (e.target.closest('.task-checkbox-container') || e.target.closest('.task-delete')) {
        return;
      }
      
      if (task.completed) return;
      
      if (activeTaskId === task.id) {
        // Toggle off
        activeTaskId = null;
        activeTaskDisplay.textContent = "No active task selected";
      } else {
        activeTaskId = task.id;
        activeTaskDisplay.textContent = `Studying: ${task.name}`;
      }
      saveTasks();
      renderTasks();
      updateStatsDisplay();
    });

    // Handle delete task
    const delBtn = li.querySelector(".task-delete");
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteTask(task.id);
    });

    tasksList.appendChild(li);
  });
}

function deleteTask(id) {
  tasks = tasks.filter(t => t.id !== id);
  if (activeTaskId === id) {
    activeTaskId = null;
    activeTaskDisplay.textContent = "No active task selected";
  }
  saveTasks();
  renderTasks();
  updateStatsDisplay();
}

function saveTasks() {
  localStorage.setItem("focus_study_tasks", JSON.stringify(tasks));
  localStorage.setItem("active_task_id", activeTaskId || "");
}

function loadTasks() {
  const data = localStorage.getItem("focus_study_tasks");
  if (data) {
    tasks = JSON.parse(data);
  }
  const savedActiveTaskId = localStorage.getItem("active_task_id");
  if (savedActiveTaskId && tasks.some(t => t.id === savedActiveTaskId)) {
    activeTaskId = savedActiveTaskId;
    const activeTask = tasks.find(t => t.id === activeTaskId);
    if (activeTask) {
      activeTaskDisplay.textContent = `Studying: ${activeTask.name}`;
    }
  } else {
    activeTaskId = null;
    activeTaskDisplay.textContent = "No active task selected";
  }
  renderTasks();
}

// Convert 24hr string "23:15" to "11:15 PM"
function format12h(timeStr) {
  const [h, m] = timeStr.split(":").map(Number);
  const ampm = h >= 12 ? "PM" : "AM";
  const displayH = h % 12 === 0 ? 12 : h % 12;
  return `${displayH}:${String(m).padStart(2, "0")} ${ampm}`;
}

// ==========================================
// AI OVERLAY DRAWING & UI STATUS LOGS
// ==========================================
function updateLogs(facePresent, personDetected, gazeStatus, phoneDetected, detectedLabels, handsCount) {
  if (facePresent) {
    logPresence.textContent = "Present";
    logPresence.className = "status-ok";
  } else if (personDetected) {
    logPresence.textContent = "Present (Desk)";
    logPresence.className = "status-ok";
  } else {
    logPresence.textContent = "Absent";
    logPresence.className = "status-error";
  }

  if (isCalibrated) {
    if (gazeStatus === "center") {
      logGaze.textContent = "Screen Focus";
      logGaze.className = "status-ok";
    } else if (gazeStatus === "book") {
      logGaze.textContent = "Reading Book";
      logGaze.className = "status-ok";
    } else if (gazeStatus === "down_no_book") {
      logGaze.textContent = "Down (No Book)";
      logGaze.className = "status-error";
    } else {
      logGaze.textContent = "Looking Away";
      logGaze.className = "status-error";
    }
  } else {
    logGaze.textContent = "Uncalibrated";
    logGaze.className = "status-warn";
  }

  if (phoneDetected) {
    logPhone.textContent = "Phone Detected";
    logPhone.className = "status-error";
  } else {
    logPhone.textContent = "No Distraction";
    logPhone.className = "status-ok";
  }

  // Update hands tracked status
  if (handsCount > 0) {
    logHands.textContent = `${handsCount} Hand${handsCount > 1 ? "s" : ""} Tracked`;
    logHands.className = "status-ok";
  } else {
    logHands.textContent = "No Hands Visible";
    logHands.className = "";
  }

  // Update live list of detected non-person objects
  if (detectedLabels && detectedLabels.length > 0) {
    logObjects.textContent = detectedLabels.join(", ");
    logObjects.className = "status-ok";
  } else {
    logObjects.textContent = "None";
    logObjects.className = "";
  }
}

function drawHandOutline(handLandmarks) {
  canvasCtx.strokeStyle = isDistracted ? "rgba(239, 68, 68, 0.45)" : "rgba(59, 130, 246, 0.55)";
  canvasCtx.fillStyle = isDistracted ? "rgba(239, 68, 68, 0.6)" : "rgba(59, 130, 246, 0.7)";
  canvasCtx.lineWidth = 2.5;

  // Connection indices for hands
  const connections = [
    [0, 1, 2, 3, 4],       // Thumb
    [0, 5, 6, 7, 8],       // Index
    [9, 10, 11, 12],       // Middle
    [13, 14, 15, 16],      // Ring
    [0, 17, 18, 19, 20],   // Pinky
    [5, 9, 13, 17]         // Palm base
  ];

  connections.forEach(chain => {
    canvasCtx.beginPath();
    chain.forEach((idx, i) => {
      const pt = handLandmarks[idx];
      const px = pt.x * canvasElement.width;
      const py = pt.y * canvasElement.height;
      if (i === 0) {
        canvasCtx.moveTo(px, py);
      } else {
        canvasCtx.lineTo(px, py);
      }
    });
    canvasCtx.stroke();
  });

  // Draw landmarks as circles
  handLandmarks.forEach(pt => {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x * canvasElement.width, pt.y * canvasElement.height, 4, 0, 2 * Math.PI);
    canvasCtx.fill();
  });
}

function drawFaceOutline(landmarks) {
  // Set draw styles
  canvasCtx.strokeStyle = isDistracted ? "rgba(239, 68, 68, 0.4)" : "rgba(16, 185, 129, 0.4)";
  canvasCtx.lineWidth = 1.5;
  
  // Draw an oval outline around key face contour coordinates (simplified outline)
  // Contour indices: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  // We can just connect a few points or draw lines between connections
  // Let's connect landmarks[33] to landmarks[263] for eyes, landmarks[4] nose, landmarks[152] chin to visualize
  canvasCtx.beginPath();
  
  // Left eye
  const le = landmarks[33];
  canvasCtx.arc(le.x * canvasElement.width, le.y * canvasElement.height, 4, 0, 2 * Math.PI);
  // Right eye
  const re = landmarks[263];
  canvasCtx.arc(re.x * canvasElement.width, re.y * canvasElement.height, 4, 0, 2 * Math.PI);
  
  // Chin
  const ch = landmarks[152];
  canvasCtx.arc(ch.x * canvasElement.width, ch.y * canvasElement.height, 4, 0, 2 * Math.PI);
  
  // Nose
  const ns = landmarks[4];
  canvasCtx.arc(ns.x * canvasElement.width, ns.y * canvasElement.height, 4, 0, 2 * Math.PI);
  
  canvasCtx.fillStyle = isDistracted ? "rgba(239, 68, 68, 0.6)" : "rgba(16, 185, 129, 0.6)";
  canvasCtx.fill();
}

function drawGazeDot(diffX, diffY, bookDetected) {
  // Draw boundaries lines in center of camera panel (canvas) for vertical tracking
  const centerW = canvasElement.width / 2;
  const centerH = canvasElement.height / 2;
  const boxH_down = 0.45 * canvasElement.height * 1.5;
  
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  canvasCtx.lineWidth = 1;
  canvasCtx.setLineDash([4, 4]);

  // Lower limit line (only visible and relevant if strict book verification is enabled)
  if (requireBookInFrame) {
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, centerH + boxH_down);
    canvasCtx.lineTo(canvasElement.width, centerH + boxH_down);
    canvasCtx.stroke();
  }
  canvasCtx.setLineDash([]);

  // Gaze target circle (center)
  canvasCtx.strokeStyle = (isDistracted || isGazeDistracted) ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)";
  canvasCtx.beginPath();
  canvasCtx.arc(centerW, centerH, 15, 0, 2 * Math.PI);
  canvasCtx.stroke();

  // Position visual gaze dot pointer
  gazeDot.classList.remove("hidden");
  
  // Interpolate position - map X and Y to percentage space
  // We use YAW_THRESHOLD * 4 to allow a wider visual space for horizontal movement of the dot
  const dotX = 50 + (diffX / (YAW_THRESHOLD * 4)) * 50;
  const dotY = 50 + (diffY / (0.45 * 2)) * 50; // Visual scale factor
  
  // Clamp boundaries to [5, 95]
  const clampedX = Math.max(5, Math.min(95, dotX));
  const clampedY = Math.max(5, Math.min(95, dotY));
  
  gazeDot.style.left = `${clampedX}%`;
  gazeDot.style.top = `${clampedY}%`;

  let isPitchAway = false;
  // Only evaluate pitch away (looking down without book) if strict checking is enabled
  if (requireBookInFrame && diffY > PITCH_THRESHOLD && !bookDetected) {
    isPitchAway = true;
  }

  if (isPitchAway) {
    gazeDot.style.backgroundColor = "var(--warning-glow)";
    gazeDot.style.boxShadow = "0 0 10px var(--warning-glow)";
  } else {
    gazeDot.style.backgroundColor = "var(--primary-glow)";
    gazeDot.style.boxShadow = "0 0 10px var(--primary-glow)";
  }
}

function drawObjectBox(box, label, color) {
  const x = box.originX;
  const y = box.originY;
  const w = box.width;
  const h = box.height;

  let strokeColor, fillColor;
  if (color === "red") {
    strokeColor = "rgba(239, 68, 68, 0.85)";
    fillColor = "rgba(239, 68, 68, 0.15)";
  } else if (color === "green") {
    strokeColor = "rgba(16, 185, 129, 0.85)";
    fillColor = "rgba(16, 185, 129, 0.15)";
  } else {
    strokeColor = "rgba(59, 130, 246, 0.85)";
    fillColor = "rgba(59, 130, 246, 0.15)";
  }

  canvasCtx.strokeStyle = strokeColor;
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeRect(x, y, w, h);

  canvasCtx.fillStyle = fillColor;
  canvasCtx.fillRect(x, y, w, h);

  // Label background
  canvasCtx.fillStyle = strokeColor;
  canvasCtx.font = "bold 12px sans-serif";
  const labelText = `${label}`;
  const textWidth = canvasCtx.measureText(labelText).width;
  canvasCtx.fillRect(x, y - 20, textWidth + 10, 20);

  // Label text
  canvasCtx.fillStyle = "#fff";
  canvasCtx.fillText(labelText, x + 5, y - 5);
}

// ==========================================
// BANGLADESH STANDARD TIME DIGITAL CLOCK
// ==========================================
function updateDigitalClock() {
  const clockElement = document.getElementById("digital-clock");
  if (!clockElement) return;

  const now = new Date();
  
  // Format options explicitly to Bangladesh Standard Time (BST) timezone
  const options = {
    timeZone: 'Asia/Dhaka',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true
  };
  
  const formatter = new Intl.DateTimeFormat('en-US', options);
  clockElement.textContent = formatter.format(now);

  // Auto-update task start time to current time if the user hasn't edited it manually and isn't currently editing it.
  if (!isStartTimeManuallyEdited && document.activeElement !== taskStartTimeInput) {
    setDefaultStartTime();
  }

  // Extract hour, minute, second for analog clock in Dhaka timezone
  try {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Dhaka',
      hour: 'numeric',
      minute: 'numeric',
      second: 'numeric',
      hour12: false
    }).formatToParts(now);

    let hr = 0, min = 0, sec = 0;
    parts.forEach(part => {
      if (part.type === 'hour') hr = parseInt(part.value);
      if (part.type === 'minute') min = parseInt(part.value);
      if (part.type === 'second') sec = parseInt(part.value);
    });

    // Calculate angles
    const secAngle = sec * 6; // 360 / 60
    const minAngle = min * 6 + sec * 0.1; // 360 / 60
    const hrAngle = (hr % 12) * 30 + min * 0.5; // 360 / 12

    // Update analog hands
    const hrHand = document.getElementById("analog-hour");
    const minHand = document.getElementById("analog-minute");
    const secHand = document.getElementById("analog-second");

    if (hrHand) hrHand.style.transform = `translate(-50%, 0) rotate(${hrAngle}deg)`;
    if (minHand) minHand.style.transform = `translate(-50%, 0) rotate(${minAngle}deg)`;
    if (secHand) secHand.style.transform = `translate(-50%, 0) rotate(${secAngle}deg)`;
  } catch (err) {
    console.warn("Failed to update analog clock hands:", err);
  }
}

// Start clock interval
setInterval(updateDigitalClock, 1000);
updateDigitalClock(); // Update immediately on load

function recoverSession() {
  const sessionRunning = localStorage.getItem("study_session_running") === "true";
  if (sessionRunning) {
    const endTime = parseInt(localStorage.getItem("study_session_end_time"));
    const duration = parseInt(localStorage.getItem("study_session_duration"));
    const timeLeft = Math.round((endTime - Date.now()) / 1000);
    
    if (timeLeft > 0) {
      // Restore stats
      totalFocusedSeconds = parseInt(localStorage.getItem("study_session_total_focused")) || 0;
      totalLossSeconds = parseInt(localStorage.getItem("study_session_total_loss")) || 0;
      totalSessionSeconds = parseInt(localStorage.getItem("study_session_total_session")) || 0;
      distractionCount = parseInt(localStorage.getItem("study_session_distractions")) || 0;
      
      sessionDuration = duration;
      timeRemaining = timeLeft;
      
      // Calculate time lost during page reload / tab closure
      const expectedTimeLeft = duration - totalSessionSeconds;
      const reloadLoss = Math.max(0, expectedTimeLeft - timeLeft);
      
      if (reloadLoss > 0) {
        totalLossSeconds += reloadLoss;
        totalSessionSeconds += reloadLoss;
        
        // Also apply it to the active task if selected
        if (activeTaskId) {
          const activeTask = tasks.find(t => t.id === activeTaskId);
          if (activeTask) {
            if (!activeTask.lossSeconds) activeTask.lossSeconds = 0;
            if (!activeTask.sessionSeconds) activeTask.sessionSeconds = 0;
            activeTask.lossSeconds += reloadLoss;
            activeTask.sessionSeconds += reloadLoss;
            saveTasks();
            const statsEl = document.getElementById(`task-stats-${activeTask.id}`);
            if (statsEl) {
              statsEl.textContent = `🎯 Focused: ${formatSeconds(activeTask.focusedSeconds || 0)} | 📉 Loss: ${formatSeconds(activeTask.lossSeconds || 0)} | ⚠️ Distractions: ${activeTask.distractionCount || 0}`;
            }
          }
        }
      }
      
      updateTimerDisplay();
      updateStatsDisplay();
      
      // Auto-start the timer and enter distraction state until they calibrate/fullscreen
      isTimerRunning = true;
      btnStart.classList.add("hidden");
      
      btnPause.classList.add("hidden");
      btnReset.classList.add("hidden");
      btnRestart.classList.add("hidden");
      document.body.classList.add("lockdown-active");
      
      presetButtons.forEach(btn => btn.disabled = true);
      customMinutesInput.disabled = true;
      
      // Show ETA (Estimated finishing time)
      const timerEta = document.getElementById("timer-eta");
      if (timerEta) {
        const finishTime = new Date(Date.now() + timeRemaining * 1000);
        timerEta.textContent = `Ends at: ${getDhakaFormattedETA(finishTime)}`;
        timerEta.classList.remove("hidden");
      }
      
      startTimerInterval();
      
      // Force distraction overlay
      triggerDistraction(0, 0, "reload");
    } else {
      localStorage.removeItem("study_session_running");
    }
  }
}

// PWA Service Worker Registration (disabled on localhost to prevent local caching issues)
if ('serviceWorker' in navigator && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(err => {
      console.warn('Service worker registration failed:', err);
    });
  });
}

