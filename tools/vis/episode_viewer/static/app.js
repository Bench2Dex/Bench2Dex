const state = {
  summary: null,
  frame: null,
  contact: null,
  currentIdx: 0,
  activeTab: "rgb",
  playTimer: null,
  isPlaying: false,
  playbackSeq: 0,
  playbackSpeed: 1.0,
  bundleCache: new Map(), // idx -> bundle {frame, images:{tab:{cam:dataurl}}, tactile:{site:dataurl}}
  cachedTab: "rgb",       // tab the current bundleCache was prefetched for
  cachedTactile: false,   // whether bundles in cache include tactile data urls
  prefetchSeq: 0,         // bumped to cancel running prefetch loops
  prefetchProgress: { loaded: 0, total: 0 },
  debounceTimer: null,
  episodeRequestSeq: 0,
  frameRequestSeq: 0,
  entries: [],
  episodes: [],
  currentEpisodeIndex: 0,
  currentDirectory: "",
  currentFile: null,
  canGoUp: false,
  parentDirectory: null,
  cameraTiles: {},
  tactileTiles: {},
};

const objectColors = [
  "#e69f00",
  "#56b4e9",
  "#009e73",
  "#0072b2",
  "#d55e00",
  "#cc79a7",
  "#6f6f6f",
];

document.addEventListener("DOMContentLoaded", init);
window.addEventListener("resize", () => {
  drawTopdown();
  drawSeries();
  drawTactileTimelines();
});

async function init() {
  bindControls();
  try {
    await loadEpisodeList();
    if (state.currentFile) {
      await loadCurrentEpisodeByPath(state.currentFile, { selectBackend: false });
    } else {
      showNoEpisodeSelected("Open a folder and select an HDF5 episode.");
    }
  } catch (err) {
    document.body.innerHTML = `<div class="empty-note">Failed to load episode summary: ${escapeHtml(err.message)}</div>`;
    return;
  }
}

function bindControls() {
  document.getElementById("episodeSelect").addEventListener("change", async (event) => {
    const index = parseInt(event.target.value || "0", 10);
    try {
      await loadCurrentEpisode(index);
    } catch (err) {
      document.getElementById("filePath").textContent = `episode load failed: ${err.message}`;
    }
  });
  document.getElementById("upDirBtn").addEventListener("click", async () => {
    if (!state.canGoUp) return;
    try {
      await openDirectory(state.parentDirectory || "");
    } catch (err) {
      document.getElementById("filePath").textContent = `directory open failed: ${err.message}`;
    }
  });
  document.getElementById("deleteEntryBtn").addEventListener("click", async () => {
    if (!state.currentFile) return;
    try {
      await deleteEpisodeEntry(state.currentFile);
    } catch (err) {
      document.getElementById("filePath").textContent = `delete failed: ${err.message}`;
    }
  });
  document.getElementById("episodeBrowser").addEventListener("click", async (event) => {
    const deleteButton = event.target.closest(".browser-entry-delete");
    if (deleteButton) {
      try {
        await deleteEpisodeEntry(deleteButton.dataset.entryPath);
      } catch (err) {
        document.getElementById("filePath").textContent = `delete failed: ${err.message}`;
      }
      return;
    }

    const mainButton = event.target.closest(".browser-entry-main");
    if (!mainButton) return;
    try {
      if (mainButton.dataset.entryType === "directory") {
        await openDirectory(mainButton.dataset.entryPath);
      } else {
        await loadCurrentEpisodeByPath(mainButton.dataset.entryPath, { autoPlay: true });
      }
    } catch (err) {
      document.getElementById("filePath").textContent = `episode load failed: ${err.message}`;
    }
  });
  document.getElementById("prevBtn").addEventListener("click", () => setFrame(state.currentIdx - 1));
  document.getElementById("nextBtn").addEventListener("click", () => setFrame(state.currentIdx + 1));
  document.getElementById("frameInput").addEventListener("change", (event) => {
    setFrame(parseInt(event.target.value || "0", 10));
  });
  document.getElementById("frameSlider").addEventListener("input", (event) => {
    const idx = parseInt(event.target.value || "0", 10);
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => setFrame(idx), 60);
  });
  document.getElementById("playBtn").addEventListener("click", togglePlay);
  const speedInput = document.getElementById("speedInput");
  if (speedInput) {
    const _applySpeed = () => {
      const v = parseFloat(speedInput.value);
      state.playbackSpeed = Number.isFinite(v) && v > 0 ? Math.min(20, Math.max(0.1, v)) : 1.0;
      speedInput.value = String(state.playbackSpeed);
    };
    speedInput.addEventListener("change", _applySpeed);
    speedInput.addEventListener("input", _applySpeed);
  }
  document.getElementById("jumpContactBtn").addEventListener("click", jumpToNextContact);
  document.getElementById("tabs").addEventListener("click", (event) => {
    const button = event.target.closest(".tab-button");
    if (!button) return;
    const previousTab = state.activeTab;
    state.activeTab = button.dataset.tab;
    for (const tab of document.querySelectorAll(".tab-button")) {
      tab.classList.toggle("active", tab === button);
    }
    if (state.activeTab !== previousTab) {
      // The bundleCache is keyed per tab; on tab change, drop stale entries
      // and re-fetch + restart prefetch for the new tab.
      state.bundleCache = new Map();
      state.cachedTab = state.activeTab;
      state.prefetchSeq += 1;
      state.prefetchProgress = { loaded: 0, total: state.summary ? state.summary.frame_count || 0 : 0 };
      updatePrefetchChip();
      setFrame(state.currentIdx);
      if (state.summary) startEpisodePrefetch(state.episodeRequestSeq);
    } else {
      updateCameraImages();
    }
  });
}

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadEpisodeList() {
  const payload = await fetchJson("api/episodes");
  applyEpisodeListPayload(payload);
}

function applyEpisodeListPayload(payload) {
  if (Object.prototype.hasOwnProperty.call(payload, "entries")) {
    state.entries = payload.entries || [];
  }
  if (Object.prototype.hasOwnProperty.call(payload, "episodes")) {
    state.episodes = payload.episodes || [];
  }
  if (Number.isFinite(payload.current_index)) {
    state.currentEpisodeIndex = payload.current_index;
  }
  if (Object.prototype.hasOwnProperty.call(payload, "current_file")) {
    state.currentFile = payload.current_file || null;
  }
  if (Object.prototype.hasOwnProperty.call(payload, "current_directory")) {
    state.currentDirectory = payload.current_directory || "";
  }
  if (Object.prototype.hasOwnProperty.call(payload, "can_go_up")) {
    state.canGoUp = Boolean(payload.can_go_up);
  }
  if (Object.prototype.hasOwnProperty.call(payload, "parent_directory")) {
    state.parentDirectory = payload.parent_directory;
  }
  setupEpisodeSelect();
  renderEpisodeBrowser();
}

function setupEpisodeSelect() {
  const select = document.getElementById("episodeSelect");
  const episodes = state.episodes || [];
  select.innerHTML = episodes
    .map((episode, idx) => {
      const index = Number.isFinite(episode.index) ? episode.index : idx;
      return `<option value="${index}" data-path="${escapeHtml(episode.path || "")}">${escapeHtml(episode.file_name || episode.name)}</option>`;
    })
    .join("");
  select.value = String(state.currentEpisodeIndex || 0);
  select.disabled = episodes.length <= 1;
}

function renderEpisodeBrowser() {
  const browser = document.getElementById("episodeBrowser");
  if (!browser) return;
  const entries = state.entries || [];
  const currentDir = state.currentDirectory ? state.currentDirectory : "/";
  document.getElementById("currentDirLabel").textContent = currentDir;
  document.getElementById("upDirBtn").disabled = !state.canGoUp;
  document.getElementById("deleteEntryBtn").disabled = !state.currentFile;

  if (!entries.length) {
    browser.innerHTML = `<div class="browser-empty">No HDF5 files or folders here.</div>`;
    return;
  }

  browser.innerHTML = entries
    .map((entry) => {
      const isCurrent = entry.type === "file" && entry.path === state.currentFile;
      const disabledDelete = entry.deletable === false ? " disabled" : "";
      return `
        <div class="browser-entry ${entry.type}${isCurrent ? " active" : ""}" data-entry-type="${escapeHtml(entry.type)}" data-entry-path="${escapeHtml(entry.path)}">
          <button class="browser-entry-main" data-entry-type="${escapeHtml(entry.type)}" data-entry-path="${escapeHtml(entry.path)}" title="${escapeHtml(entry.path)}">
            <span class="entry-kind">${entry.type === "directory" ? "DIR" : "HDF5"}</span>
            <span class="entry-name">${escapeHtml(entry.name || entry.file_name || entry.path)}</span>
          </button>
          <button class="browser-entry-delete" data-entry-type="${escapeHtml(entry.type)}" data-entry-path="${escapeHtml(entry.path)}" title="Move to deleted_items"${disabledDelete}>Delete</button>
        </div>
      `;
    })
    .join("");
}

async function loadCurrentEpisode(index, options = {}) {
  return loadEpisodeBySelection({ index }, options);
}

async function loadCurrentEpisodeByPath(path, options = {}) {
  return loadEpisodeBySelection({ path }, options);
}

async function loadEpisodeBySelection(selection, options = {}) {
  stopPlayback();
  const requestSeq = state.episodeRequestSeq + 1;
  state.episodeRequestSeq = requestSeq;
  state.frameRequestSeq += 1;

  if (options.selectBackend !== false) {
    const body = Object.prototype.hasOwnProperty.call(selection, "path")
      ? { path: selection.path }
      : { index: Number.isFinite(selection.index) ? selection.index : state.currentEpisodeIndex || 0 };
    const selected = await fetchJson("api/episode/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (requestSeq !== state.episodeRequestSeq) return;
    applyEpisodeListPayload(selected);
  }

  state.summary = null;
  state.frame = null;
  state.contact = null;
  state.currentIdx = 0;
  state.cameraTiles = {};
  state.tactileTiles = {};

  const summary = await fetchJson("api/summary");
  if (requestSeq !== state.episodeRequestSeq) return;
  state.summary = summary;
  setupSummary();
  buildCameraGrid();
  buildTactileGrid();
  // New episode -> drop bundle cache and bump prefetch seq to cancel any
  // in-flight prefetch loop from the previous episode.
  state.bundleCache = new Map();
  state.prefetchSeq += 1;
  state.prefetchProgress = { loaded: 0, total: summary.frame_count || 0 };
  state.cachedTab = state.activeTab;
  state.cachedTactile = (state.summary.tactile_sites || []).length > 0;
  updatePrefetchChip();
  await Promise.all([loadContactSummary(requestSeq), setFrame(0)]);
  if (requestSeq === state.episodeRequestSeq) {
    // Kick off background prefetch of the whole episode into bundleCache so
    // that subsequent setFrame() calls hit memory and playback is smooth.
    startEpisodePrefetch(requestSeq);
    if (options.autoPlay && (summary.frame_count || 0) > 0) {
      togglePlay();
    }
  }
}

async function openDirectory(path) {
  stopPlayback();
  const payload = await fetchJson("api/directory/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: path || "" }),
  });
  applyEpisodeListPayload(payload);
  if (state.currentFile) {
    await loadCurrentEpisodeByPath(state.currentFile, { selectBackend: false });
  } else {
    showNoEpisodeSelected("Open a folder and select an HDF5 episode.");
  }
}

async function deleteEpisodeEntry(path, options = {}) {
  if (!path) return;
  const entry = (state.entries || []).find((item) => item.path === path);
  const label = entry ? `${entry.type} ${entry.name || entry.file_name || entry.path}` : path;
  if (!options.skipConfirm) {
    // Custom confirm dialog — avoids Chrome's "prevent additional dialogs"
    // checkbox that permanently blocks window.confirm().
    const ok = await _customConfirm(`Move ${label} to deleted_items?`);
    if (!ok) return;
  }
  stopPlayback();
  const result = await fetchJson("api/episode/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  applyEpisodeListPayload(result.browser || {});
  if (state.currentFile) {
    await loadCurrentEpisodeByPath(state.currentFile, { selectBackend: false });
    document.getElementById("filePath").textContent = `moved to ${result.deleted_path}`;
  } else {
    showNoEpisodeSelected(`Moved to ${result.deleted_path}`);
  }
}

function showNoEpisodeSelected(message) {
  state.summary = null;
  state.frame = null;
  state.contact = null;
  state.currentIdx = 0;
  state.cameraTiles = {};
  state.tactileTiles = {};
  document.getElementById("fileName").textContent = "No HDF5 selected";
  document.getElementById("filePath").textContent = message || "";
  document.getElementById("fpsChip").textContent = "0 FPS";
  document.getElementById("frameInput").value = "0";
  document.getElementById("frameInput").max = "0";
  document.getElementById("frameSlider").value = "0";
  document.getElementById("frameSlider").max = "0";
  document.getElementById("episodeSummary").innerHTML = `<div class="empty-note">Select an HDF5 episode.</div>`;
  document.getElementById("cameraGrid").innerHTML = `<div class="empty-note">No episode selected.</div>`;
  document.getElementById("objectTables").innerHTML = "";
  document.getElementById("robotTables").innerHTML = "";
  document.getElementById("tactileGrid").innerHTML = "";
  document.getElementById("tactileStatus").textContent = "no episode selected";
  drawTopdown();
  drawSeries();
}

function setupSummary() {
  const meta = state.summary.metadata || {};
  const frameMax = Math.max(0, (state.summary.frame_count || 1) - 1);
  document.getElementById("fileName").textContent = state.summary.file_name || "episode";
  document.getElementById("filePath").textContent = state.summary.path || "";
  document.getElementById("fpsChip").textContent = `${meta.fps || 0} FPS`;
  document.getElementById("frameInput").max = String(frameMax);
  document.getElementById("frameSlider").max = String(frameMax);

  const rows = [
    ["Instruction", meta.instruction],
    ["Scene", meta.scene_name],
    ["Task", meta.task_name],
    ["Robot", meta.robot_key],
    ["FPS", meta.fps],
    ["Frames", state.summary.frame_count],
    ["Schema", meta.schema_version],
    ["Success", formatBool(meta.success)],
    ["Valid Frames", state.summary.valid_frame_count],
    ["Cameras", `${state.summary.camera_ids.length} (${state.summary.camera_ids.join(", ")})`],
    ["Tactile Sites", `${state.summary.tactile_sites.length}`],
    ["Objects", `${state.summary.object_ids.length}`],
    ["Box Labels", formatBoxLabelSummary()],
    ["Occupancy", formatOccupancySummary()],
  ];
  for (const item of state.summary.generalization || []) {
    rows.push([item.label, item.value]);
  }

  const container = document.getElementById("episodeSummary");
  container.innerHTML = rows
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `<div class="summary-key">${escapeHtml(key)}</div><div class="summary-value">${escapeHtml(String(value))}</div>`)
    .join("");
}

function buildCameraGrid() {
  const grid = document.getElementById("cameraGrid");
  grid.innerHTML = "";
  state.cameraTiles = {};
  const cameraIds = state.summary.camera_ids || [];
  if (!cameraIds.length) {
    grid.innerHTML = `<div class="empty-note">No cameras in this episode.</div>`;
    return;
  }
  for (const cam of cameraIds) {
    const tile = document.createElement("article");
    tile.className = "camera-tile";
    tile.innerHTML = `
      <div class="camera-title">
        <span class="primary"></span>
        <span class="secondary"></span>
      </div>
      <img alt="${escapeHtml(cam)}">
    `;
    grid.appendChild(tile);
    state.cameraTiles[cam] = {
      tile,
      primary: tile.querySelector(".primary"),
      secondary: tile.querySelector(".secondary"),
      img: tile.querySelector("img"),
    };
  }
}

function buildTactileGrid() {
  const grid = document.getElementById("tactileGrid");
  grid.innerHTML = "";
  state.tactileTiles = {};
  const sites = state.summary.tactile_sites || [];
  if (!sites.length) {
    grid.innerHTML = `<div class="empty-note">No tactile TacMap data in this episode.</div>`;
    return;
  }

  for (const side of ["left", "right"]) {
    const column = document.createElement("section");
    column.className = "hand-column";
    column.innerHTML = `<h3 class="hand-title">${side === "left" ? "Left Hand" : "Right Hand"}</h3>`;
    for (const site of sites.filter((name) => name.startsWith(`${side}_`))) {
      const item = document.createElement("article");
      item.className = "tactile-site";
      item.innerHTML = `
        <div class="site-name" title="${escapeHtml(site)}">${escapeHtml(site.replace(`${side}_`, ""))}</div>
        <img alt="${escapeHtml(site)}">
        <canvas class="timeline"></canvas>
      `;
      column.appendChild(item);
      state.tactileTiles[site] = {
        img: item.querySelector("img"),
        canvas: item.querySelector("canvas"),
      };
    }
    grid.appendChild(column);
  }
}

async function loadContactSummary(requestSeq = state.episodeRequestSeq) {
  try {
    const contact = await fetchJson("api/tactile/contact_summary");
    if (requestSeq !== state.episodeRequestSeq) return;
    state.contact = contact;
    drawTactileTimelines();
    drawSeries();
  } catch (err) {
    document.getElementById("tactileStatus").textContent = `contact summary failed: ${err.message}`;
  }
}

async function setFrame(rawIdx) {
  const frameCount = state.summary ? state.summary.frame_count : 1;
  const maxIdx = Math.max(0, frameCount - 1);
  const idx = Math.max(0, Math.min(maxIdx, Number.isFinite(rawIdx) ? rawIdx : 0));
  const requestSeq = state.frameRequestSeq + 1;
  state.frameRequestSeq = requestSeq;
  state.currentIdx = idx;
  document.getElementById("frameInput").value = String(idx);
  document.getElementById("frameSlider").value = String(idx);
  try {
    const bundle = await getOrFetchBundle(idx);
    if (requestSeq !== state.frameRequestSeq || state.currentIdx !== idx) {
      return;
    }
    state.frame = bundle.frame;
    state.frameImages = bundle.images || {};
    state.frameTactile = bundle.tactile || {};
  } catch (err) {
    if (requestSeq !== state.frameRequestSeq) {
      return;
    }
    console.error(err);
    return;
  }
  renderFrame();
}

async function getOrFetchBundle(idx) {
  const cached = state.bundleCache.get(idx);
  // Cache hit only valid if the active tab is the one we cached.
  if (cached && state.activeTab === state.cachedTab) {
    return cached;
  }
  const tabsParam = encodeURIComponent(state.activeTab);
  const wantsTactile = (state.summary.tactile_sites || []).length > 0 ? 1 : 0;
  const bundle = await fetchJson(`api/frame_bundle/${idx}?tabs=${tabsParam}&include_tactile=${wantsTactile}`);
  // Store in cache only if the tab is unchanged from when we cached the rest.
  if (state.activeTab === state.cachedTab) {
    state.bundleCache.set(idx, bundle);
  } else {
    // Tab changed - reset cache to be tab-consistent.
    state.bundleCache = new Map([[idx, bundle]]);
    state.cachedTab = state.activeTab;
    state.prefetchSeq += 1;
    state.prefetchProgress = { loaded: 1, total: state.summary.frame_count || 1 };
    updatePrefetchChip();
    // Restart prefetch for the new tab in the background.
    startEpisodePrefetch(state.episodeRequestSeq);
  }
  return bundle;
}

async function startEpisodePrefetch(episodeSeq) {
  if (!state.summary) return;
  const prefetchSeq = ++state.prefetchSeq;
  const total = state.summary.frame_count || 0;
  state.prefetchProgress = { loaded: state.bundleCache.size, total };
  updatePrefetchChip();

  const tabsParam = encodeURIComponent(state.activeTab);
  const wantsTactile = (state.summary.tactile_sites || []).length > 0 ? 1 : 0;

  // Sequential fetch keeps the server single-threaded h5py happy and avoids
  // queueing up too much memory. Browser already keeps connection warm.
  for (let i = 0; i < total; i++) {
    if (prefetchSeq !== state.prefetchSeq || episodeSeq !== state.episodeRequestSeq) return;
    if (state.bundleCache.has(i)) {
      state.prefetchProgress.loaded += 0; // already counted via cache.size
      continue;
    }
    try {
      const bundle = await fetchJson(`api/frame_bundle/${i}?tabs=${tabsParam}&include_tactile=${wantsTactile}`);
      if (prefetchSeq !== state.prefetchSeq || episodeSeq !== state.episodeRequestSeq) return;
      state.bundleCache.set(i, bundle);
    } catch (err) {
      console.warn(`prefetch frame ${i} failed`, err);
    }
    state.prefetchProgress.loaded = state.bundleCache.size;
    if (i % 5 === 0 || i === total - 1) updatePrefetchChip();
  }
  updatePrefetchChip();
}

function updatePrefetchChip() {
  const chip = document.getElementById("prefetchChip");
  if (!chip) return;
  const { loaded, total } = state.prefetchProgress;
  if (!total) {
    chip.textContent = "";
    chip.style.display = "none";
    return;
  }
  chip.style.display = "";
  if (loaded >= total) {
    chip.textContent = `cached ${total}/${total}`;
  } else {
    chip.textContent = `caching ${loaded}/${total}`;
  }
}

function renderFrame() {
  updateStatusChips();
  updateCameraImages();
  renderObjects();
  renderRobotTables();
  updateTactileImages();
  drawTopdown();
  drawSeries();
}

function updateStatusChips() {
  const status = state.frame.status || {};
  setChip("validChip", status.frame_valid, "valid", "invalid");
  setChip("actionChip", status.action_valid, "action valid", "action invalid");
  const success = status.success;
  const done = status.done || status.is_last;
  const chip = document.getElementById("successChip");
  chip.textContent = `${done ? "done" : "running"} / ${formatBool(success)}`;
  chip.className = "status-chip";
  chip.classList.add(success === true ? "ok" : success === false ? "bad" : "warn");
}

function setChip(id, value, trueText, falseText) {
  const chip = document.getElementById(id);
  chip.textContent = value === true ? trueText : value === false ? falseText : "unknown";
  chip.className = "status-chip";
  chip.classList.add(value === true ? "ok" : value === false ? "bad" : "warn");
}

function updateCameraImages() {
  if (!state.frame || !state.summary) return;
  const frameIdx = framePayloadIndex();
  for (const cam of state.summary.camera_ids || []) {
    const refs = state.cameraTiles[cam];
    if (!refs) continue;
    const info = (state.summary.camera_info || {})[cam] || {};
    const range = ((state.frame.depth_ranges || {})[cam]) || {};
    refs.primary.textContent = `${cam} · ${info.camera_type || "camera"}`;
    if (state.activeTab === "occupancy") {
      refs.secondary.textContent = formatOccupancyCounts();
    } else {
      refs.secondary.textContent = state.activeTab.startsWith("box") ? formatBoxCounts(cam) : formatDepthRange(range);
    }
    refs.img.src = (state.frameImages && state.frameImages[state.activeTab] && state.frameImages[state.activeTab][cam])
      || cameraImageUrl(cam, frameIdx);
  }
}

function cameraImageUrl(cam, idx) {
  const encoded = encodeURIComponent(cam);
  if (state.activeTab === "depth") return `/image/depth/${encoded}/${idx}.png`;
  if (state.activeTab === "occupancy") return `/image/occupancy/${encoded}/${idx}.png`;
  if (state.activeTab === "box2d") return `/image/box/box2d/${encoded}/${idx}.png`;
  if (state.activeTab === "box3d") return `/image/box/box3d/${encoded}/${idx}.png`;
  return `/image/rgb/${encoded}/${idx}`;
}

function formatBoxLabelSummary() {
  if (!state.summary) return "";
  const box3dCount = (state.summary.box3d_object_ids || []).length;
  const box2dCams = (state.summary.box2d_camera_ids || []).length;
  const parts = [];
  parts.push(`box3d ${box3dCount} objects`);
  parts.push(`box2d ${box2dCams} cameras`);
  return parts.join(", ");
}

function formatOccupancySummary() {
  const info = state.summary ? state.summary.occupancy_info || {} : {};
  if (!info.available) return "missing";
  const shape = (info.grid_shape || []).join("x");
  const voxel = info.voxel_size ? `${fmt(info.voxel_size)} m` : "unknown voxel";
  const label = info.label_key || "occupancy";
  const semantic = info.has_semantic_id ? "semantic" : "state";
  return `${label} ${shape} · ${voxel} · ${semantic}`;
}

function formatOccupancyCounts() {
  const occ = state.frame ? state.frame.occupancy_current || {} : {};
  if (!occ.available) return "occupancy missing";
  const valid = occ.frame_valid === false ? " · invalid" : "";
  const semantic = (occ.semantic_counts || [])
    .filter((item) => item.count > 0)
    .slice(0, 2)
    .map((item) => `${compactObjectId(item.label)} ${item.count}`)
    .join(" · ");
  const suffix = semantic ? ` · ${semantic}` : "";
  return `occ ${occ.occupied || 0} / ${occ.total || 0}${valid}${suffix}`;
}

function formatBoxCounts(cam) {
  const counts = state.frame && state.frame.box_counts ? state.frame.box_counts[cam] : null;
  if (!counts) return "box labels missing";
  return `2D visible ${counts.box2d_visible || 0} · 3D objects ${counts.box3d_objects || 0}`;
}

function formatDepthRange(range) {
  if (!range || range.available === false) return "depth missing";
  if (!range.finite) return "depth no finite";
  const invalid = range.invalid ? ` · invalid ${range.invalid}` : "";
  return `p1-p99 ${fmt(range.p1)}-${fmt(range.p99)} m${invalid}`;
}

function renderObjects() {
  const container = document.getElementById("objectTables");
  const objects = state.frame.objects || [];
  if (!objects.length) {
    container.innerHTML = `<div class="empty-note">No object state.</div>`;
    return;
  }
  const dynamic = objects.filter((obj) => obj.body_type !== "articulation" && !obj.articulation.length);
  const articulation = objects.filter((obj) => obj.body_type === "articulation" || obj.articulation.length);
  container.innerHTML = renderObjectTable("Dynamic Objects", dynamic) + renderObjectTable("Articulations", articulation);
}

function renderObjectTable(title, objects) {
  if (!objects.length) return "";
  const rows = objects
    .map((obj) => {
      const joints = obj.articulation && obj.articulation.length
        ? obj.articulation.map((joint) => `${escapeHtml(joint.name)}=${fmt(joint.qpos)}`).join("<br>")
        : "";
      return `
        <tr>
          <td title="${escapeHtml(obj.id)}">${escapeHtml(compactObjectId(obj.id))}</td>
          <td>${escapeHtml(obj.role || obj.body_type || "")}</td>
          <td>${vecText(obj.xyz)}</td>
          <td>${vecText(obj.quat_xyzw)}</td>
          <td>${fmt(speed(obj.lin_vel))}</td>
          <td>${joints}</td>
        </tr>
      `;
    })
    .join("");
  return `
    <div class="table-title">${escapeHtml(title)}</div>
    <table>
      <thead><tr><th>object</th><th>role</th><th>xyz</th><th>quat</th><th>speed</th><th>joints</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderRobotTables() {
  const container = document.getElementById("robotTables");
  const groups = (state.frame.robot && state.frame.robot.groups) || [];
  if (!groups.length) {
    container.innerHTML = `<div class="empty-note">No robot joint state.</div>`;
    return;
  }
  container.innerHTML = groups
    .map((group) => {
      const rows = group.rows
        .map(
          (row) => `
            <tr>
              <td>${row.index}</td>
              <td title="${escapeHtml(row.full_name)}">${escapeHtml(row.name)}</td>
              <td>${fmt(row.qpos)}</td>
              <td>${fmt(row.qvel)}</td>
              <td>${fmt(row.qeffort)}</td>
              <td>${fmt(row.action)}</td>
            </tr>
          `
        )
        .join("");
      const open = group.id === "left_arm" || group.id === "right_arm" ? "open" : "";
      return `
        <details ${open}>
          <summary>${escapeHtml(group.label)} · ${group.rows.length}</summary>
          <table>
            <thead><tr><th>#</th><th>joint</th><th>qpos</th><th>qvel</th><th>qeffort</th><th>action</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>
      `;
    })
    .join("");
}

function updateTactileImages() {
  if (!state.summary || !state.frame) return;
  const frameIdx = framePayloadIndex();
  const activeSites = [];
  for (const site of state.summary.tactile_sites || []) {
    const refs = state.tactileTiles[site];
    if (!refs) continue;
    refs.img.src = (state.frameTactile && state.frameTactile[site])
      || `/image/tactile/${encodeURIComponent(site)}/${frameIdx}.png`;
    const siteState = state.frame.tactile_current && state.frame.tactile_current.sites
      ? state.frame.tactile_current.sites[site]
      : null;
    if (siteState && siteState.nonzero) activeSites.push(`${site} (${siteState.max})`);
  }
  const status = document.getElementById("tactileStatus");
  status.textContent = activeSites.length ? activeSites.join(", ") : "no contact at this frame";
  drawTactileTimelines();
}

function framePayloadIndex() {
  if (state.frame && Number.isFinite(state.frame.idx)) {
    return state.frame.idx;
  }
  return state.currentIdx;
}

function togglePlay() {
  const button = document.getElementById("playBtn");
  if (state.isPlaying) {
    stopPlayback();
    return;
  }
  if (!state.summary || !state.summary.frame_count) return;
  state.isPlaying = true;
  state.playbackSeq += 1;
  button.textContent = "Pause";
  schedulePlaybackTick(0, state.playbackSeq);
}

function playbackIntervalMs() {
  const fps = Math.max(1, Math.min(60, (state.summary.metadata && state.summary.metadata.fps) || 20));
  // For speed > 1x we advance multiple frames per tick (see playbackTick) and
  // keep the per-tick interval at native rate, since the fetch round-trip is
  // the real bottleneck. For speed < 1x we slow the interval down directly.
  const speed = state.playbackSpeed > 0 ? state.playbackSpeed : 1.0;
  const subSpeed = speed >= 1.0 ? 1.0 : speed;
  return 1000 / (fps * subSpeed);
}

function playbackStride() {
  const speed = state.playbackSpeed > 0 ? state.playbackSpeed : 1.0;
  // 2x -> stride 2, 3x -> 3, 1.5x -> 1 (sub-1 speeds use interval slowdown).
  return Math.max(1, Math.floor(speed));
}

function playbackNowMs() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function schedulePlaybackTick(delayMs, playbackSeq) {
  if (!state.isPlaying || playbackSeq !== state.playbackSeq) return;
  state.playTimer = setTimeout(() => playbackTick(playbackSeq), Math.max(0, delayMs));
}

async function playbackTick(playbackSeq) {
  state.playTimer = null;
  if (!state.isPlaying || playbackSeq !== state.playbackSeq || !state.summary) return;
  const stride = playbackStride();
  const total = state.summary.frame_count;
  let next = state.currentIdx + stride;
  if (next >= total) next = next % total;
  const startedAt = playbackNowMs();
  await setFrame(next);
  if (!state.isPlaying || playbackSeq !== state.playbackSeq) return;
  const elapsedMs = playbackNowMs() - startedAt;
  schedulePlaybackTick(playbackIntervalMs() - elapsedMs, playbackSeq);
}

function stopPlayback() {
  if (!state.isPlaying && !state.playTimer) return;
  state.isPlaying = false;
  state.playbackSeq += 1;
  if (state.playTimer) {
    clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  const button = document.getElementById("playBtn");
  if (button) button.textContent = "Play";
}

function jumpToNextContact() {
  const frames = state.contact && state.contact.total ? state.contact.total.nonzero_frames || [] : [];
  if (!frames.length) return;
  const next = frames.find((frame) => frame > state.currentIdx);
  setFrame(next === undefined ? frames[0] : next);
}

function drawTactileTimelines() {
  if (!state.contact) return;
  for (const [site, refs] of Object.entries(state.tactileTiles)) {
    const siteContact = state.contact.sites ? state.contact.sites[site] : null;
    drawTimeline(refs.canvas, siteContact ? siteContact.max_values : [], state.currentIdx, "#d55e00", 255);
  }
}

function drawTimeline(canvas, values, frame, color, maxValue) {
  const ctx = prepareCanvas(canvas);
  if (!ctx) return;
  const width = canvas.clientWidth || 200;
  const height = canvas.clientHeight || 28;
  ctx.fillStyle = "#fafbf8";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9ddd6";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  if (!values || !values.length) return;
  const max = Math.max(1, maxValue || Math.max(...values));
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const idx = Math.min(values.length - 1, Math.round((x / Math.max(1, width - 1)) * (values.length - 1)));
    const y = height - 4 - (values[idx] / max) * (height - 8);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const fx = (frame / Math.max(1, values.length - 1)) * width;
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(fx, 0);
  ctx.lineTo(fx, height);
  ctx.stroke();
}

function drawSeries() {
  const canvas = document.getElementById("seriesCanvas");
  const ctx = prepareCanvas(canvas);
  if (!ctx || !state.frame) return;
  const width = canvas.clientWidth || 320;
  const height = canvas.clientHeight || 100;
  ctx.fillStyle = "#fbfcf9";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9ddd6";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  const series = state.frame.series || {};
  const frames = series.frames || [];
  drawLineSeries(ctx, frames, series.qpos_mean_abs || [], width, height, "#0072b2");
  drawLineSeries(ctx, frames, series.action_mean_abs || [], width, height, "#e69f00");
  if (state.contact && state.contact.total && frames.length) {
    const values = frames.map((frame) => state.contact.total.max_values[frame] || 0);
    drawLineSeries(ctx, frames, values, width, height, "#d55e00", 255);
  }

  if (frames.length) {
    const fx = ((state.currentIdx - frames[0]) / Math.max(1, frames[frames.length - 1] - frames[0])) * width;
    ctx.strokeStyle = "#111";
    ctx.beginPath();
    ctx.moveTo(fx, 0);
    ctx.lineTo(fx, height);
    ctx.stroke();
  }
  ctx.fillStyle = "#1e2423";
  ctx.fillText("qpos mean abs", 8, 14);
  ctx.fillStyle = "#9b5b16";
  ctx.fillText("action mean abs", 8, 30);
  ctx.fillStyle = "#b33a31";
  ctx.fillText("contact max", 8, 46);
}

function drawLineSeries(ctx, frames, values, width, height, color, fixedMax) {
  if (!frames.length || !values.length) return;
  const validValues = values.filter((value) => value !== null && Number.isFinite(value));
  if (!validValues.length) return;
  const max = fixedMax || Math.max(...validValues, 1e-6);
  const minFrame = frames[0];
  const maxFrame = frames[frames.length - 1];
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < frames.length; i += 1) {
    const value = values[i];
    if (value === null || !Number.isFinite(value)) continue;
    const x = ((frames[i] - minFrame) / Math.max(1, maxFrame - minFrame)) * width;
    const y = height - 8 - (value / max) * (height - 18);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

function drawTopdown() {
  const canvas = document.getElementById("topdownCanvas");
  const ctx = prepareCanvas(canvas);
  if (!ctx || !state.summary) return;
  const width = canvas.clientWidth || 320;
  const height = canvas.clientHeight || 180;
  ctx.fillStyle = "#fbfcf9";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9ddd6";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  const bounds = expandedBounds(state.summary.track_bounds);
  if (!bounds) {
    ctx.fillStyle = "#68716d";
    ctx.fillText("No object tracks", 12, 24);
    return;
  }

  drawGrid(ctx, width, height);
  const tracks = state.summary.object_tracks || {};
  const objectIds = state.summary.object_ids || Object.keys(tracks);
  objectIds.forEach((objectId, idx) => {
    const track = tracks[objectId];
    if (!track || !track.xy || !track.xy.length) return;
    ctx.beginPath();
    track.xy.forEach((point, pointIdx) => {
      const [x, y] = mapXY(point[0], point[1], bounds, width, height);
      if (pointIdx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = objectColors[idx % objectColors.length];
    ctx.lineWidth = 1.4;
    ctx.stroke();
  });

  const currentObjects = state.frame ? state.frame.objects || [] : [];
  currentObjects.forEach((obj, idx) => {
    if (!obj.xyz || obj.xyz.length < 2) return;
    const [x, y] = mapXY(obj.xyz[0], obj.xyz[1], bounds, width, height);
    ctx.fillStyle = objectColors[idx % objectColors.length];
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#1e2423";
    ctx.fillText(compactObjectId(obj.id), x + 6, y - 5);
  });
}

function expandedBounds(bounds) {
  if (!bounds || bounds.xmin === null || bounds.xmax === null || bounds.ymin === null || bounds.ymax === null) {
    return null;
  }
  let xmin = Number(bounds.xmin);
  let xmax = Number(bounds.xmax);
  let ymin = Number(bounds.ymin);
  let ymax = Number(bounds.ymax);
  if (![xmin, xmax, ymin, ymax].every(Number.isFinite)) return null;
  if (xmax <= xmin) xmax = xmin + 1;
  if (ymax <= ymin) ymax = ymin + 1;
  const padX = Math.max(0.05, (xmax - xmin) * 0.12);
  const padY = Math.max(0.05, (ymax - ymin) * 0.12);
  return { xmin: xmin - padX, xmax: xmax + padX, ymin: ymin - padY, ymax: ymax + padY };
}

function mapXY(x, y, bounds, width, height) {
  const px = ((x - bounds.xmin) / (bounds.xmax - bounds.xmin)) * (width - 24) + 12;
  const py = height - (((y - bounds.ymin) / (bounds.ymax - bounds.ymin)) * (height - 24) + 12);
  return [px, py];
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "#edf0eb";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = (width * i) / 4;
    const y = (height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function prepareCanvas(canvas) {
  if (!canvas) return null;
  if (typeof canvas.getBoundingClientRect !== "function" || typeof canvas.getContext !== "function") return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function compactObjectId(id) {
  return String(id || "").replace(/^obj_/, "").replace(/_/g, " ");
}

function vecText(values) {
  if (!values || !values.length) return "";
  return values.map((value) => fmt(value)).join(", ");
}

function speed(values) {
  if (!values || values.length < 3 || values.some((value) => value === null)) return null;
  return Math.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

function fmt(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  return Number(value).toFixed(3);
}

function formatBool(value) {
  if (value === true) return "true";
  if (value === false) return "false";
  return "unknown";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/** Custom confirm dialog — avoids Chrome permanently blocking window.confirm(). */
function _customConfirm(message) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center";
    const box = document.createElement("div");
    box.style.cssText = "background:#1e1e1e;border:1px solid #444;border-radius:8px;padding:24px;max-width:420px;color:#ccc;font:14px sans-serif";
    box.innerHTML = `<p style="margin:0 0 16px">${message}</p>
      <div style="display:flex;gap:8px;justify-content:flex-end">
        <button id="_cfm_cancel" style="padding:6px 16px;border:1px solid #555;border-radius:4px;background:#333;color:#ccc;cursor:pointer">Cancel</button>
        <button id="_cfm_ok" style="padding:6px 16px;border:none;border-radius:4px;background:#d9534f;color:#fff;cursor:pointer">Delete</button>
      </div>`;
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    const cleanup = (result) => { document.body.removeChild(overlay); resolve(result); };
    document.getElementById("_cfm_ok").onclick = () => cleanup(true);
    document.getElementById("_cfm_cancel").onclick = () => cleanup(false);
    overlay.onclick = (e) => { if (e.target === overlay) cleanup(false); };
  });
}
