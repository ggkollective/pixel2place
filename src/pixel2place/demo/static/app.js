import { $, api, node } from "./http.js";
import { fixtureText, initOrder, shiftRequest, supplyText } from "./order.js";

const statusLabels = {
  running: "Running",
  cancelling: "Stopping",
  completed: "Released",
  aborted: "Stopped",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
  // 교대의 주문에만 있는 상태다. 시작하지 않은 주문이다.
  queued: "Queued",
  skipped: "Not run",
};
const shiftLabels = {
  finished: "Shift finished",
  cancelled: "Shift cancelled",
  aborted: "Shift stopped",
  interrupted: "Shift interrupted",
};
const dispositionLabels = {
  released: "Released",
  reworked_released: "Reworked and released",
  rejected: "Rejected",
};
// 끝난 실행은 처분으로 표시한다. 처분이 없거나 중단인 실행은 프로세스 상태로 표시한다.
const statusText = (run) =>
  dispositionLabels[run.disposition] ||
  statusLabels[run.status] ||
  run.status.replaceAll("_", " ");
let order = null,
  mode = "reference",
  variant = "after",
  // 재작업을 한 실행은 영상이 2개다. "first" 또는 "rework"다.
  recording = "first",
  selected = null,
  active = false,
  actionPending = false,
  // 가장 최근의 교대다. 없으면 null이다.
  shift = null;
let generation = 0,
  current = null,
  timer,
  polling = false;
const stageNames = [
  "prepare_layout",
  "calibrate_board",
  "execute_cell",
  "final",
];
const stateStage = {
  created: 0,
  layout_retryable: 0,
  layout_ready: 1,
  calibrated: 2,
  checked: 3,
};
const errors = {
  protocol_error:
    "The model response or saved files did not meet the run requirements.",
  model_error: "The model request failed. The cell stopped this run.",
  recording_error:
    "The cell could not save the run records. It stopped the run.",
  cli_failed:
    "The worker stopped before a final report was available. Check the server setup and local worker log.",
  server_interrupted: "The server stopped during this run. The run did not resume.",
  user_cancelled:
    "The user cancelled this run. The server stopped its worker and simulation.",
  agent_aborted: "The cell stopped the assembly run.",
  check_failed: "One or more parts did not pass the placement check.",
  inspection_rejected:
    "The camera check did not pass. The cell did not release the board.",
  plan_rejected:
    "The plan did not have the parts of the work order. The cell did not place parts.",
  budget_exhausted: "The run reached a limit for requests, data or robot operations.",
};

function formError(message) {
  $("form-error").textContent = message || "";
  $("form-error").hidden = !message;
}
const shiftRunning = () => shift?.status === "running";
function controls() {
  // 열린 교대는 주문 사이에도 셀을 쓴다. 그때는 실행 중인 작업이 없다.
  const busy = active || shiftRunning();
  $("start").disabled = busy || actionPending || !order?.ready();
  $("start").firstElementChild.textContent = actionPending
    ? "Sending…"
    : busy
      ? "Cell is busy"
      : "Send to cell";
  $("start-shift").disabled = busy || actionPending;
  $("cancel-shift").disabled = actionPending;
  $("cancel-shift").hidden = !shiftRunning();
  $("cancel").disabled = actionPending;
  $("cancel").hidden = !(
    mode === "live" &&
    current &&
    ["running", "cancelling"].includes(current.status)
  );
}
function board(layout) {
  const canvas = $("board"),
    ctx = canvas.getContext("2d"),
    size = canvas.width;
  const rows = layout?.board.rows || 8,
    cols = layout?.board.cols || 8;
  const margin = 27,
    unit = (size - 2 * margin) / Math.max(rows, cols);
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#e9ecef";
  ctx.fillRect(margin, margin, cols * unit, rows * unit);
  ctx.strokeStyle = "#d0d5db";
  ctx.lineWidth = 1;
  for (let row = 0; row <= rows; row++) {
    ctx.beginPath();
    ctx.moveTo(margin, margin + row * unit);
    ctx.lineTo(margin + cols * unit, margin + row * unit);
    ctx.stroke();
  }
  for (let col = 0; col <= cols; col++) {
    ctx.beginPath();
    ctx.moveTo(margin + col * unit, margin);
    ctx.lineTo(margin + col * unit, margin + rows * unit);
    ctx.stroke();
  }
  ctx.fillStyle = "#8a95a1";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i < cols; i++)
    ctx.fillText(String(i), margin + (i + 0.5) * unit, 15);
  for (let i = 0; i < rows; i++)
    ctx.fillText(String(i), 12, margin + (i + 0.6) * unit);
  const colors = {
    blue: "#4f86bd",
    gray: "#7b868f",
    orange: "#e4974a",
    white: "#fbfbf7",
  };
  for (const part of layout?.parts || []) {
    ctx.fillStyle = colors[part.color] || "#9aa4ae";
    ctx.strokeStyle = part.color === "white" ? "#b9bfc6" : "#00000010";
    for (const [row, col] of part.cells) {
      ctx.beginPath();
      ctx.roundRect(
        margin + col * unit + 3,
        margin + row * unit + 3,
        unit - 6,
        unit - 6,
        4,
      );
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#00000014";
      ctx.beginPath();
      ctx.arc(
        margin + (col + 0.5) * unit,
        margin + (row + 0.5) * unit,
        2.5,
        0,
        2 * Math.PI,
      );
      ctx.fill();
      ctx.fillStyle = colors[part.color] || "#9aa4ae";
    }
  }
  $("board-size").textContent = `${rows} × ${cols}`;
  $("board-caption").textContent = layout
    ? `${layout.parts.length} parts · ${layout.board.cell_size_mm} mm pitch · Planned target layout`
    : "The planned layout appears here.";
}
function media(url, poster, caption, still = null) {
  const video = $("video");
  if (video.dataset.source !== (url || "")) {
    video.pause();
    video.dataset.source = url || "";
    if (url) video.src = url;
    else video.removeAttribute("src");
    if (poster) video.poster = poster;
    else video.removeAttribute("poster");
    video.load();
  }
  video.hidden = !url;
  $("observation").hidden = Boolean(url) || !still;
  if (still) $("observation").src = still;
  else $("observation").removeAttribute("src");
  $("video-empty").hidden = Boolean(url || still);
  $("video-caption").textContent = caption;
}
function checks(check) {
  const parts = $("parts");
  parts.replaceChildren();
  $("passed-count").textContent = check
    ? `${check.passed_count}/${check.parts.length}`
    : "—";
  $("passed-count").style.color =
    check && !check.passed ? "var(--danger)" : "var(--ok)";
  if (!check) {
    const row = node("tr"),
      cell = node(
        "td",
        "Inspection results appear after assembly.",
        "empty-table",
      );
    cell.colSpan = 5;
    row.append(cell);
    parts.append(row);
    return;
  }
  for (const part of check.parts) {
    const row = node("tr");
    row.append(
      node("td", part.part_id),
      node("td", part.type.replaceAll("_", " ")),
      node("td", `${part.position_error_mm.toFixed(2)} mm`),
      node("td", `${part.yaw_error_deg.toFixed(2)}°`),
    );
    const verdict = node("td"),
      pill = node(
        "span",
        part.passed ? "OK" : "NG",
        part.passed ? "passed-pill" : "failed-pill",
      );
    if (!part.passed) pill.title = part.failures.join(", ");
    verdict.append(pill);
    row.append(verdict);
    parts.append(row);
  }
}
const partCount = (count) => `${count} ${count === 1 ? "part" : "parts"}`;
// 카메라 판정은 최종 이미지에서 읽은 기록이다. 통과 기준은 부품 판정 표다.
function cameraCheck(report) {
  const line = $("camera-check");
  line.hidden = !report;
  if (!report) {
    line.textContent = "";
    return;
  }
  const matched = report.match_count,
    mismatched = report.mismatch_count,
    shadowed = report.shadowed_count,
    unexpected = report.unexpected_cells.length;
  const sentences = [
    `Camera check: ${matched} of ${partCount(report.parts.length)} ${matched === 1 ? "matches" : "match"} the plan.`,
  ];
  if (mismatched)
    sentences.push(
      `${partCount(mismatched)} ${mismatched === 1 ? "does" : "do"} not match.`,
    );
  // 그림자 상태는 색을 확인하지 못한 부품이다. 팔 그림자 안의 부품과 기운 부품이다.
  if (shadowed)
    sentences.push(
      `The camera cannot confirm the color of ${partCount(shadowed)}. The cause can be the arm shadow or a tilted part.`,
    );
  if (unexpected)
    sentences.push(
      `${unexpected} ${unexpected === 1 ? "cell that is not in the plan is" : "cells that are not in the plan are"} not empty.`,
    );
  line.className = `camera-check ${!report.passed ? "different" : shadowed ? "unconfirmed" : "same"}`;
  line.textContent = sentences.join(" ");
}
// 재작업 호출의 진행이다. 호출이 없으면 null이다. 결과가 오기 전에는 "active"다.
function reworkProgress(view) {
  const events = (view?.events || []).filter(
    (event) => event.call?.name === "rework_cell",
  );
  if (!events.length) return null;
  const result = events.find((event) => event.kind === "tool_result");
  return !result
    ? "active"
    : result.result?.status === "ok"
      ? "done"
      : "failed";
}
function pipeline(view) {
  const done = new Set();
  for (const event of view?.events || []) {
    if (event.kind === "tool_result" && event.result?.status === "ok")
      done.add(event.call.name);
  }
  let state = view?.runtime_state;
  if (view && stateStage[state] === undefined && state !== "completed") {
    const last = [...view.events]
      .reverse()
      .find((event) => stateStage[event.from_state] !== undefined);
    state = last?.from_state || "created";
  }
  const rework = reworkProgress(view);
  // 재작업은 배치와 판정을 다시 한다. 재작업의 결과가 오기 전에는 그 단계가 현재 단계다.
  const index = !view
    ? -1
    : rework === "active" || rework === "failed"
      ? stageNames.indexOf("execute_cell")
      : (stateStage[state] ?? 3);
  for (const [i, name] of stageNames.entries()) {
    const stage = document.querySelector(`[data-stage="${name}"]`);
    let status = done.has(name) ? "done" : "";
    if (view?.status === "completed") status = "done";
    else if (
      view &&
      i === index &&
      ["running", "cancelling"].includes(view.status)
    )
      status = "active";
    else if (
      view &&
      i === index &&
      ["aborted", "cancelled", "interrupted"].includes(view.status)
    )
      status = "failed";
    stage.className = `stage ${status}`;
    let label = {
      done: "Done",
      active: "In progress",
      failed: "Stopped",
      "": "Waiting",
    }[status];
    if (name === "execute_cell" && rework && status === "active")
      label = "Rework in progress";
    else if (name === "execute_cell" && rework === "done" && status === "done")
      label = "Reworked";
    else if (name === "final" && status === "done") label = "Released";
    else if (
      name === "final" &&
      status === "failed" &&
      view.disposition === "rejected"
    )
      label = "Rejected";
    stage.querySelector(".stage-status").textContent = label;
  }
}
// 처분은 끝난 실행의 출하, 재작업 뒤 출하 또는 불량이다. 중단한 실행에는 줄이 없다.
function disposition(view) {
  const line = $("disposition");
  const label = dispositionLabels[view?.disposition];
  line.hidden = !label;
  if (!label) {
    line.textContent = "";
    return;
  }
  const rework = view.events.find(
    (event) =>
      event.kind === "tool_result" &&
      event.call?.name === "rework_cell" &&
      event.result?.status === "ok",
  );
  const sentences = [`Disposition: ${label}.`];
  if (rework)
    sentences.push(
      `The cell placed ${partCount(rework.call.arguments.part_ids.length)} again.`,
    );
  if (rework && view.disposition === "rejected")
    sentences.push("The board did not pass the second inspection.");
  line.className = `disposition ${view.disposition}`;
  line.textContent = sentences.join(" ");
}
// 재작업을 한 실행의 표와 카메라 판정은 재작업 뒤의 판정이다. 이 줄은 재작업 앞의 판정이다.
function firstInspection(view) {
  const line = $("first-inspection");
  const check = view?.first_check,
    camera = view?.first_image_check;
  line.hidden = !check;
  if (!check) {
    line.textContent = "";
    return;
  }
  const sentences = [
    `Before the rework: ${check.passed_count} of ${partCount(check.parts.length)} OK.`,
  ];
  if (camera)
    sentences.push(
      `The camera check found ${partCount(camera.mismatch_count)} that did not match the plan.`,
    );
  line.textContent = sentences.join(" ");
}
function duration(value) {
  if (value === null || value === undefined) return "—";
  const total = Math.round(value);
  return total >= 60
    ? `${Math.floor(total / 60)} min ${total % 60} s`
    : `${total} s`;
}
// 통과율은 주문의 수와 함께 쓴다. `total`은 지표가 있는 주문의 수다.
const yieldText = (count, total) =>
  `${count} of ${total} ${total === 1 ? "order" : "orders"} (${Math.round((100 * count) / total)}%)`;
// 현장 요청 카드의 주문은 카드의 제목으로 표시한다. 다른 주문은 이름으로 표시한다.
const requestTitle = (name) =>
  document.querySelector(`.scenario[data-scenario="${name}"] strong`)
    ?.textContent || name;
function shiftRow(item, index) {
  const row = node(
    "tr",
    undefined,
    mode === "live" && item.run_id && item.run_id === selected ? "selected" : "",
  );
  const order = node("td"),
    title = `${index + 1} · ${requestTitle(item.name)}`,
    report = item.report;
  if (item.run_id) {
    const open = node("button", title, "link-button");
    open.type = "button";
    open.addEventListener("click", () => select(item.run_id));
    order.append(open);
  } else order.append(node("strong", title));
  order.append(
    node(
      "small",
      [
        `${item.order_kind === "kit" ? "Kit list" : "Board order"} · ${partCount(item.ordered_parts)}`,
        fixtureText(item.fixture),
        supplyText(item.faults),
      ]
        .filter(Boolean)
        .join(" · "),
    ),
  );
  row.append(
    order,
    node(
      "td",
      statusText(item),
      `shift-state ${item.disposition || item.status}`,
    ),
    node(
      "td",
      report && report.passed_parts !== null
        ? `${report.passed_parts}/${report.part_count}`
        : "—",
    ),
    node("td", duration(report?.robot_time_s)),
    node("td", duration(report?.wall_time_s)),
    node("td", report ? `${report.model_time_s.toFixed(1)} s` : "—"),
    node(
      "td",
      !report
        ? "—"
        : report.request_retries
          ? `${report.nemotron_requests} · ${report.request_retries} sent again`
          : String(report.nemotron_requests),
    ),
  );
  return row;
}
// 교대의 표는 읽기 전용이다. 주문의 줄은 그 주문의 작업을 연다.
function shiftPanel() {
  $("shift-panel").hidden = !shift;
  if (!shift) return;
  const running = shiftRunning(),
    started = shift.items.filter((item) => item.run_id).length,
    measured = shift.items.filter((item) => item.report).length,
    totals = shift.totals;
  $("shift-status").textContent = running
    ? `Order ${started} of ${shift.items.length}`
    : shiftLabels[shift.status];
  $("shift-status").className =
    `status-badge ${running ? "running" : shift.status === "finished" ? "completed" : "aborted"}`;
  $("shift-items").replaceChildren(...shift.items.map(shiftRow));
  $("shift-totals").textContent = measured
    ? [
        `First-pass yield: ${yieldText(totals.first_pass_orders, measured)}.`,
        `Final yield: ${yieldText(totals.completed_orders, measured)}.`,
        `Changeovers: ${totals.changeovers}.`,
        `Wall time: ${duration(totals.wall_time_s)}.`,
      ].join(" ")
    : running
      ? "The metrics appear when the first order ends."
      : "This shift has no metrics.";
}
function decisions(view) {
  const events = (view?.events || []).filter(
    (event) => event.kind === "model_result",
  );
  $("decision-count").textContent = events.length
    ? `${events.length} model decisions`
    : "";
  const container = $("decisions");
  container.replaceChildren();
  if (!events.length) {
    container.append(
      node(
        "p",
        "Send a work order to see the recorded model decisions.",
        "empty-copy",
      ),
    );
    return;
  }
  for (const event of events) {
    const item = node("div", undefined, "decision-item");
    item.append(
      node("span", `#${event.inference}`),
      node(
        "strong",
        event.call?.name || event.final?.decision || "Rejected response",
      ),
      node(
        "span",
        event.error_code ||
          event.final?.reason_code ||
          // 재계획은 모델이 고른 사유를 보여 준다.
          event.call?.arguments.reason ||
          "Contract accepted",
      ),
      node("span", event.model.split("/").pop()),
    );
    container.append(item);
  }
}
function live(view) {
  if (mode !== "live") return;
  current = view;
  $("reference-controls").hidden = true;
  $("live-tab").classList.add("active");
  $("reference-tab").classList.remove("active");
  $("media-label").textContent = "THIS JOB";
  $("run-status").textContent = view ? statusText(view) : "Ready";
  $("run-status").className = `status-badge ${view?.status || ""}`;
  $("result-title").textContent = view?.check
    ? "Inspection complete."
    : "Every part is inspected before release.";
  $("result-description").textContent = view
    ? [
        "This job",
        fixtureText(view.fixture),
        supplyText(view.faults),
        `Job ${view.id.slice(0, 8)}`,
      ]
        .filter(Boolean)
        .join(" · ")
    : "Send a work order to inspect its assembly.";
  $("progress-copy").textContent = view
    ? view.events.length
      ? "Each stage shows the cell's real state."
      : "The cell is checking the work order."
    : "Each stage updates from the cell's real events.";
  $("run-error").hidden = !view?.error_code;
  $("run-error").textContent = view?.error_code
    ? `${errors[view.error_code] || "The job stopped before completion."} (${view.error_code})`
    : "";
  const match = view?.order_match;
  $("order-match").hidden = !match;
  $("order-match").className = `order-match ${match?.matches ? "same" : "different"}`;
  $("order-match").textContent = !match
    ? ""
    : match.matches
      ? view.order.kind === "kit"
        ? "The plan has the ordered part quantities. The planner chose the positions."
        : "The plan matches the work order."
      : `The plan differs from the work order. Missing parts: ${match.missing}. Extra parts: ${match.extra}. The cell assembles the plan.`;
  disposition(view);
  cameraCheck(view?.image_check);
  firstInspection(view);
  const base = view ? `/api/runs/${view.id}/assets/` : "";
  const observed = ["calibration", "capture", "layout"].find((name) =>
    view?.assets.includes(name),
  );
  const captions = {
    calibration: "Alignment · The camera found the board and its markers.",
    capture: "Cell camera · The board before placement.",
    layout: "Plan · The target layout image for this job.",
  };
  const reworked = Boolean(view?.assets.includes("rework_video"));
  const rework = reworked && recording === "rework";
  $("recording-controls").hidden = !reworked;
  $("first-recording").classList.toggle("active", !rework);
  $("rework-recording").classList.toggle("active", rework);
  const video = rework ? "rework_video" : "video",
    final = rework ? "rework_result" : "result";
  media(
    view?.assets.includes(video) ? base + video : null,
    view?.assets.includes(final) ? base + final : null,
    view?.assets.includes(video)
      ? rework
        ? "Original recording of the rework run."
        : reworked
          ? "Original recording of the first run."
          : "Original recording of this job."
      : observed
        ? captions[observed]
        : "The recording appears when the job ends.",
    observed ? base + observed : null,
  );
  board(view?.layout);
  checks(view?.check);
  pipeline(view);
  decisions(view);
  shiftPanel();
  controls();
}
async function reference() {
  const ticket = ++generation;
  mode = "reference";
  $("reference-controls").hidden = false;
  $("recording-controls").hidden = true;
  $("reference-tab").classList.add("active");
  $("live-tab").classList.remove("active");
  $("before-reference").classList.toggle("active", variant === "before");
  $("after-reference").classList.toggle("active", variant === "after");
  $("media-label").textContent = "REFERENCE RUN · SHIFTED FIXTURE";
  $("run-status").textContent = "Reference run";
  $("run-status").className = "status-badge";
  $("elapsed").textContent = "SO-101 ARM";
  $("progress-copy").textContent =
    "The reference run is a separate recording. Send a work order to start a new job.";
  $("run-error").hidden = true;
  $("order-match").hidden = true;
  disposition(null);
  cameraCheck(null);
  firstInspection(null);
  pipeline(null);
  decisions(null);
  shiftPanel();
  controls();
  try {
    const data = await api(`/api/reference/${variant}`);
    if (ticket !== generation || mode !== "reference") return;
    board(data.layout);
    checks(data.check);
    $("result-title").textContent =
      variant === "after"
        ? "Board found. Placement corrected. Parts OK."
        : "A shifted fixture causes rejected parts.";
    $("result-description").textContent =
      `${variant === "after" ? "With alignment" : "Without alignment"} · Recorded 16-part panel. The aligned run also has a shifted camera.`;
    media(
      `/api/reference/${variant}/assets/video`,
      `/api/reference/${variant}/assets/result`,
      "Reference recording · Separate from new jobs.",
    );
  } catch {
    if (ticket !== generation) return;
    media(
      null,
      null,
      "The reference recording is not available on this server.",
    );
    board(null);
    checks(null);
    $("result-description").textContent =
      "Send a work order to record a new job on this server.";
  }
}
async function history() {
  const runs = await api("/api/runs");
  active = runs.some((run) => ["running", "cancelling"].includes(run.status));
  const container = $("history");
  container.replaceChildren();
  if (!runs.length)
    container.append(
      node("p", "No jobs yet. Send a work order to the cell.", "empty-copy"),
    );
  for (const run of runs.slice(0, 8)) {
    const item = node("button", undefined, `history-item ${run.status}`);
    item.type = "button";
    const info = node("span");
    info.append(
      node(
        "strong",
        run.ordered_parts
          ? `${run.order_kind === "kit" ? "Kit list" : "Board order"} · ${run.ordered_parts} parts${run.intake_id ? " · read from text" : ""}`
          : run.prompt,
      ),
      node(
        "small",
        [
          statusText(run),
          fixtureText(run.fixture),
          supplyText(run.faults),
          `Job ${run.id.slice(0, 8)}`,
        ]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    item.append(node("span", undefined, "history-dot"), info);
    item.addEventListener("click", () => select(run.id));
    container.append(item);
  }
  controls();
  return runs;
}
async function shifts() {
  shift = (await api("/api/shifts"))[0] || null;
  shiftPanel();
  controls();
  // 교대의 작업을 보는 화면은 교대의 현재 주문을 따라간다.
  const runs = (shift?.items || [])
    .map((item) => item.run_id)
    .filter(Boolean);
  const current = runs.at(-1);
  if (
    shiftRunning() &&
    mode === "live" &&
    current !== selected &&
    runs.includes(selected)
  )
    await select(current);
}
async function select(id) {
  if (selected !== id) recording = "first";
  selected = id;
  mode = "live";
  generation++;
  clearTimeout(timer);
  try {
    const view = await api(`/api/runs/${id}`);
    if (selected === id) {
      current = view;
      live(current);
      formError(null);
    }
  } catch (error) {
    formError(error.message);
  }
  poll();
}
async function poll() {
  clearTimeout(timer);
  if (polling) return;
  polling = true;
  const id = selected;
  try {
    if (id) {
      const view = await api(`/api/runs/${id}`);
      if (selected === id) {
        current = view;
        live(view);
      }
    }
    await history();
    await shifts();
  } catch (error) {
    formError(`Connection paused. ${error.message}`);
  } finally {
    polling = false;
    timer = setTimeout(poll, 1800);
  }
}
$("order-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (active || actionPending || !order.ready()) return;
  actionPending = true;
  controls();
  formError(null);
  try {
    const view = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(order.body()),
    });
    selected = view.id;
    current = view;
    recording = "first";
    active = true;
    mode = "live";
    generation++;
    live(view);
    poll();
  } catch (error) {
    formError(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});
$("cancel").addEventListener("click", async () => {
  if (!selected || actionPending) return;
  actionPending = true;
  controls();
  formError(null);
  try {
    current = await api(`/api/runs/${selected}/cancel`, { method: "POST" });
    live(current);
    await history();
  } catch (error) {
    formError(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});
$("start-shift").addEventListener("click", async () => {
  if (active || shiftRunning() || actionPending) return;
  actionPending = true;
  controls();
  formError(null);
  try {
    shift = await api("/api/shifts", {
      method: "POST",
      body: JSON.stringify(shiftRequest()),
    });
    active = true;
    await select(shift.items[0].run_id);
  } catch (error) {
    formError(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});
$("cancel-shift").addEventListener("click", async () => {
  if (!shiftRunning() || actionPending) return;
  actionPending = true;
  controls();
  formError(null);
  try {
    shift = await api(`/api/shifts/${shift.id}/cancel`, { method: "POST" });
    shiftPanel();
    // 보고 있는 작업이 교대의 현재 주문이면 그 작업도 멈췄다.
    if (selected) {
      current = await api(`/api/runs/${selected}`);
      live(current);
    }
    await history();
  } catch (error) {
    formError(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});
$("reference-tab").addEventListener("click", reference);
$("live-tab").addEventListener("click", () => {
  mode = "live";
  generation++;
  live(current);
});
$("before-reference").addEventListener("click", () => {
  variant = "before";
  reference();
});
$("after-reference").addEventListener("click", () => {
  variant = "after";
  reference();
});
for (const [id, value] of [
  ["first-recording", "first"],
  ["rework-recording", "rework"],
])
  $(id).addEventListener("click", () => {
    recording = value;
    live(current);
  });
$("refresh-history").addEventListener("click", () =>
  history().catch((error) => formError(error.message)),
);
$("video").addEventListener("error", () => {
  $("video-caption").textContent =
    "The recording could not load. Refresh the view to try again.";
});
setInterval(() => {
  if (mode !== "live" || !current) return;
  const end = current.finished_at
    ? Date.parse(current.finished_at)
    : Date.now();
  const seconds = Math.max(
    0,
    Math.floor((end - Date.parse(current.created_at)) / 1000),
  );
  $("elapsed").textContent =
    `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")} elapsed`;
}, 1000);
board(null);
reference();
history()
  .then((runs) => {
    const running = runs.find((run) =>
      ["running", "cancelling"].includes(run.status),
    );
    if (running) select(running.id);
    else poll();
  })
  .catch((error) => {
    formError(error.message);
    poll();
  });
// 주문 입력을 읽기 전에는 실행 버튼을 잠근다.
controls();
order = await initOrder(controls);
if (order.error) formError(order.error);
controls();
