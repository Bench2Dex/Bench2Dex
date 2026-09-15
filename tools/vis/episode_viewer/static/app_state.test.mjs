import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function makeContext() {
  const elements = new Map();
  const getElement = (id) => {
    if (!elements.has(id)) {
      elements.set(id, {
        value: "",
        textContent: "",
        innerHTML: "",
        className: "",
        disabled: false,
        style: {},
        classList: { add() {}, toggle() {} },
        addEventListener() {},
        querySelector() {
          return null;
        },
      });
    }
    return elements.get(id);
  };

  return vm.createContext({
    console,
    document: {
      addEventListener() {},
      getElementById: getElement,
      querySelectorAll() {
        return [];
      },
    },
    window: {
      addEventListener() {},
      devicePixelRatio: 1,
      confirm() {
        return true;
      },
    },
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    encodeURIComponent,
    Math,
    Number,
    Promise,
  });
}

test("applyEpisodeListPayload renders directory and file browser entries", () => {
  const context = makeContext();
  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    applyEpisodeListPayload({
      current_index: 0,
      current_file: "episode_a.hdf5",
      current_directory: "",
      can_go_up: false,
      parent_directory: null,
      entries: [
        { type: "directory", name: "nested", path: "nested", deletable: true },
        { type: "file", name: "episode_a.hdf5", file_name: "episode_a.hdf5", path: "episode_a.hdf5", deletable: true }
      ],
      episodes: [
        { type: "file", index: 0, name: "episode_a.hdf5", file_name: "episode_a.hdf5", path: "episode_a.hdf5" }
      ]
    });
    globalThis.browserHtml = document.getElementById("episodeBrowser").innerHTML;
    globalThis.upDisabled = document.getElementById("upDirBtn").disabled;
  `, context);

  assert.match(context.browserHtml, /data-entry-type="directory"/);
  assert.match(context.browserHtml, /data-entry-path="nested"/);
  assert.match(context.browserHtml, /episode_a\.hdf5/);
  assert.equal(context.upDisabled, true);
});

test("deleteEpisodeEntry moves a path and refreshes browser state", async () => {
  const context = makeContext();
  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    state.currentFile = "episode_a.hdf5";
    state.entries = [{ type: "file", name: "episode_a.hdf5", path: "episode_a.hdf5", deletable: true }];
    globalThis.calls = [];
    fetchJson = async function (path, options) {
      calls.push({ path, options });
      if (path === "api/episode/delete") {
        return {
          deleted_path: "deleted_items/episode_a_20260528.hdf5",
          browser: {
            current_index: -1,
            current_file: null,
            current_directory: "",
            can_go_up: false,
            parent_directory: null,
            entries: [],
            episodes: []
          }
        };
      }
      throw new Error(path);
    };
    stopPlayback = function () {
      globalThis.stopped = true;
    };
  `, context);

  await vm.runInContext("deleteEpisodeEntry('episode_a.hdf5', { skipConfirm: true })", context);

  const calls = JSON.parse(vm.runInContext("JSON.stringify(calls)", context));
  assert.equal(calls[0].path, "api/episode/delete");
  assert.deepEqual(JSON.parse(calls[0].options.body), { path: "episode_a.hdf5" });
  assert.equal(context.stopped, true);
  assert.equal(vm.runInContext("state.currentFile", context), null);
  assert.match(vm.runInContext("document.getElementById('fileName').textContent", context), /No HDF5/);
});

test("setFrame ignores stale frame responses that resolve after a newer frame", async () => {
  const context = makeContext();
  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    state.summary = { frame_count: 10, camera_ids: [], tactile_sites: [], metadata: {} };
    state.currentIdx = 0;
    globalThis.renderedFrames = [];
    renderFrame = function () {
      renderedFrames.push({ currentIdx: state.currentIdx, frameIdx: state.frame && state.frame.idx });
    };
  `, context);

  const first = deferred();
  const second = deferred();
  context.requests = { 1: first, 2: second };
  vm.runInContext(`
    fetchJson = function (path) {
      const idx = Number(path.split("/")[2].split("?")[0]);
      return requests[idx].promise;
    };
  `, context);

  const firstSet = vm.runInContext("setFrame(1)", context);
  const secondSet = vm.runInContext("setFrame(2)", context);

  second.resolve({ frame: { idx: 2 } });
  await secondSet;
  first.resolve({ frame: { idx: 1 } });
  await firstSet;

  const rendered = JSON.parse(vm.runInContext("JSON.stringify(renderedFrames)", context));
  assert.deepEqual(rendered, [{ currentIdx: 2, frameIdx: 2 }]);
});

test("tactile image URLs are based on the loaded frame payload", () => {
  const context = makeContext();
  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    const image = { src: "" };
    state.summary = { tactile_sites: ["right_index_2"] };
    state.frame = {
      idx: 1,
      tactile_current: { sites: { right_index_2: { max: 7, nonzero: true } } },
    };
    state.currentIdx = 2;
    state.tactileTiles = { right_index_2: { img: image, canvas: {} } };
    drawTactileTimelines = function () {};
    updateTactileImages();
    globalThis.tactileSrc = image.src;
  `, context);

  assert.equal(
    context.tactileSrc,
    "/image/tactile/right_index_2/1.png",
  );
});

test("loadCurrentEpisode switches backend episode and resets viewer state", async () => {
  const context = makeContext();
  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    state.summary = { frame_count: 8, camera_ids: ["old_cam"], tactile_sites: ["old_site"], metadata: {} };
    state.frame = { idx: 7 };
    state.contact = { stale: true };
    state.currentIdx = 7;
    state.cameraTiles = { old_cam: {} };
    state.tactileTiles = { old_site: {} };
    globalThis.calls = [];
    globalThis.renderedFrames = [];
    fetchJson = async function (path, options) {
      calls.push({ path, options });
      if (path === "api/episode/select") return { current_index: 1 };
      if (path === "api/summary") {
        return { file_name: "episode_b.hdf5", frame_count: 3, camera_ids: [], tactile_sites: [], metadata: {} };
      }
      if (path === "api/tactile/contact_summary") return { total: { max_values: [], nonzero_frames: [] }, sites: {} };
      if (path.startsWith("api/frame_bundle/")) {
        const idx = Number(path.split("/")[2].split("?")[0]);
        return { frame: { idx, status: {}, objects: [], robot: { groups: [] } } };
      }
      throw new Error(path);
    };
    stopPlayback = function () {
      globalThis.stopped = true;
    };
    setupSummary = function () {
      globalThis.loadedFileName = state.summary.file_name;
    };
    buildCameraGrid = function () {
      state.cameraTiles = {};
    };
    buildTactileGrid = function () {
      state.tactileTiles = {};
    };
    drawTactileTimelines = function () {};
    drawSeries = function () {};
    renderFrame = function () {
      renderedFrames.push(state.frame.idx);
    };
  `, context);

  await vm.runInContext("loadCurrentEpisode(1)", context);

  const calls = JSON.parse(vm.runInContext("JSON.stringify(calls)", context));
  assert.equal(calls[0].path, "api/episode/select");
  assert.deepEqual(JSON.parse(calls[0].options.body), { index: 1 });
  assert.equal(calls[1].path, "api/summary");
  assert.equal(context.loadedFileName, "episode_b.hdf5");
  assert.equal(context.stopped, true);
  assert.deepEqual(JSON.parse(vm.runInContext("JSON.stringify(renderedFrames)", context)), [0]);
  assert.equal(vm.runInContext("state.currentIdx", context), 0);
  assert.equal(vm.runInContext("state.contact.total.max_values.length", context), 0);
});

test("playback waits for each frame request before scheduling the next frame", async () => {
  const context = makeContext();
  context.scheduledTimers = [];
  vm.runInContext(`
    setTimeout = function (fn, delay) {
      const id = scheduledTimers.length + 1;
      scheduledTimers.push({ id, fn, delay, active: true, interval: false });
      return id;
    };
    clearTimeout = function (id) {
      const timer = scheduledTimers.find((item) => item.id === id);
      if (timer) timer.active = false;
    };
    setInterval = function (fn, delay) {
      const id = scheduledTimers.length + 1;
      scheduledTimers.push({ id, fn, delay, active: true, interval: true });
      return id;
    };
    clearInterval = function (id) {
      const timer = scheduledTimers.find((item) => item.id === id);
      if (timer) timer.active = false;
    };
    globalThis.runNextTimer = function () {
      const timer = scheduledTimers.find((item) => item.active);
      if (!timer) return false;
      if (!timer.interval) timer.active = false;
      timer.fn();
      return true;
    };
  `, context);

  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    state.summary = { frame_count: 5, camera_ids: [], tactile_sites: [], metadata: { fps: 20 } };
    state.currentIdx = 0;
    state.frame = { idx: 0, status: {}, objects: [], robot: { groups: [] } };
    globalThis.calls = [];
    renderFrame = function () {};
  `, context);

  const first = deferred();
  const second = deferred();
  context.requests = [first, second];
  vm.runInContext(`
    fetchJson = function (path) {
      calls.push(path);
      return requests[calls.length - 1].promise;
    };
  `, context);

  vm.runInContext("togglePlay()", context);
  assert.equal(vm.runInContext("runNextTimer()", context), true);
  assert.deepEqual(
    JSON.parse(vm.runInContext("JSON.stringify(calls)", context)),
    ["api/frame_bundle/1?tabs=rgb&include_tactile=0"],
  );

  vm.runInContext("runNextTimer()", context);
  assert.deepEqual(
    JSON.parse(vm.runInContext("JSON.stringify(calls)", context)),
    ["api/frame_bundle/1?tabs=rgb&include_tactile=0"],
  );

  first.resolve({ frame: { idx: 1, status: {}, objects: [], robot: { groups: [] } } });
  await new Promise((done) => setImmediate(done));

  assert.equal(vm.runInContext("runNextTimer()", context), true);
  assert.deepEqual(
    JSON.parse(vm.runInContext("JSON.stringify(calls)", context)),
    [
      "api/frame_bundle/1?tabs=rgb&include_tactile=0",
      "api/frame_bundle/2?tabs=rgb&include_tactile=0",
    ],
  );

  vm.runInContext("stopPlayback()", context);
  second.resolve({ frame: { idx: 2, status: {}, objects: [], robot: { groups: [] } } });
});

test("playback subtracts frame load time from the next scheduled delay", async () => {
  const context = makeContext();
  context.scheduledTimers = [];
  vm.runInContext(`
    globalThis.currentNow = 0;
    globalThis.performance = { now: function () { return currentNow; } };
    setTimeout = function (fn, delay) {
      const id = scheduledTimers.length + 1;
      scheduledTimers.push({ id, fn, delay, active: true });
      return id;
    };
    clearTimeout = function (id) {
      const timer = scheduledTimers.find((item) => item.id === id);
      if (timer) timer.active = false;
    };
    globalThis.runNextTimer = function () {
      const timer = scheduledTimers.find((item) => item.active);
      if (!timer) return false;
      timer.active = false;
      timer.fn();
      return true;
    };
  `, context);

  const source = readFileSync(new URL("./app.js", import.meta.url), "utf8");
  vm.runInContext(source, context, { filename: "app.js" });

  vm.runInContext(`
    state.summary = { frame_count: 5, camera_ids: [], tactile_sites: [], metadata: { fps: 20 } };
    state.currentIdx = 0;
    state.frame = { idx: 0, status: {}, objects: [], robot: { groups: [] } };
    renderFrame = function () {};
  `, context);

  const first = deferred();
  context.requests = [first];
  vm.runInContext(`
    fetchJson = function () {
      return requests[0].promise;
    };
  `, context);

  vm.runInContext("togglePlay()", context);
  assert.equal(vm.runInContext("runNextTimer()", context), true);

  vm.runInContext("currentNow = 60", context);
  first.resolve({ frame: { idx: 1, status: {}, objects: [], robot: { groups: [] } } });
  await new Promise((done) => setImmediate(done));

  assert.equal(vm.runInContext("scheduledTimers[scheduledTimers.length - 1].delay", context), 0);

  vm.runInContext("stopPlayback()", context);
});
