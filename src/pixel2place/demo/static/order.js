// 작업 지시 입력. 보드 편집기, 키트 목록, 자유 문장의 주문 읽기, 지그 오차, 공급 결함과 지시문 미리보기를 소유한다.
// 부품 형태, 한도, 배치 방식과 지시문 문장은 서버가 준다. 이 파일에 형태 표를 두지 않는다.
// 공급 결함의 부품과 색도 서버가 주문마다 정한다. 이 파일은 결함의 종류만 고른다.
import { $, api, node } from "./http.js";

const NOMINAL = { x_mm: 0, y_mm: 0, yaw_deg: 0 };
const SHIFTED = { x_mm: 3, y_mm: 5, yaw_deg: 1.5 };
const part = (type, color, row, col) => ({
  type,
  color,
  rotation_deg: 0,
  row,
  col,
});
const SIX_PART_KIT = [
  part("square_tile", "blue", 1, 1),
  part("square_tile", "blue", 1, 6),
  part("rect_tile", "gray", 3, 1),
  part("rect_tile", "gray", 3, 5),
  part("L_tile", "orange", 5, 1),
  part("T_tile", "white", 5, 4),
];
const scenarios = {
  changeover: { parts: SIX_PART_KIT, fixture: NOMINAL },
  kitting: {
    parts: [
      part("square_tile", "blue", 1, 1),
      part("square_tile", "blue", 1, 6),
      part("square_tile", "blue", 6, 1),
      part("square_tile", "blue", 6, 6),
      part("rect_tile", "gray", 3, 3),
      part("rect_tile", "gray", 4, 3),
    ],
    fixture: NOMINAL,
  },
  mixed: {
    parts: [
      part("L_tile", "orange", 1, 1),
      part("T_tile", "white", 1, 4),
      part("rect_tile", "gray", 5, 1),
      part("square_tile", "blue", 5, 5),
    ],
    fixture: NOMINAL,
  },
  fixture: { parts: SIX_PART_KIT, fixture: SHIFTED },
};
const fixtureInputs = {
  x_mm: { id: "fixture-x", step: 0.5, unit: " mm" },
  y_mm: { id: "fixture-y", step: 0.5, unit: " mm" },
  yaw_deg: { id: "fixture-yaw", step: 0.1, unit: "°" },
};
const PREVIEW_DELAY_MS = 150;
const QUARTER_TURN = 90;
const FULL_TURN = 360;
// 실제 크기의 오차는 도식에서 보이지 않는다.
const DIAGRAM_GAIN = 5;
const DIAGRAM_BOARD_PX = 60;
const HINT =
  "Click a cell to place the part. Click a part to remove it. Press R to rotate.";
const MODES = ["board", "kit", "text"];
// 키트 탭의 처음 수량이다. 서버가 주는 키트 종류에 없는 항목은 쓰지 않는다.
const KIT_START = { square_tile: 4, rect_tile: 2 };
const READ_HINT = "Read the order to see the instruction.";
const REJECT_TEXT = {
  not_assembly_request: "The text does not ask to place parts on the board.",
  unsupported_part_type: "The order cannot have this part type.",
  unsupported_color: "The order cannot have this color.",
  too_many_parts: "The text asks for more parts than the maximum.",
  unsupported_arrangement: "The order cannot have this arrangement.",
  ambiguous_request: "A quantity or a part type is missing or not clear.",
};
const UNREAD = { state: "blocked", text: READ_HINT, note: "" };
const SUPPLY_HIDDEN =
  " Cosmos and Nemotron do not receive this value. The camera check must find the fault.";

let options = null,
  mode = "board",
  // 지시문을 만드는 탭이다. 원문 탭에서는 마지막 주문 탭을 유지한다.
  orderMode = "board",
  parts = [],
  kit = { items: [], arrangement: null },
  kitControls = [],
  tool = { type: "square_tile", color: "blue", rotation_deg: 0 },
  fixture = { ...NOMINAL },
  // 공급 결함의 종류다. "none", "missing" 또는 "wrong_color"다.
  supply = "none",
  preview = { state: "blocked", text: "" },
  // 자유 문장에서 읽은 주문이다. 문장을 고치면 버린다.
  reading = UNREAD,
  readTicket = 0,
  ticket = 0,
  timer,
  notify = () => {};

const signed = (value, digits) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;

export function fixtureText(value) {
  if (!value.x_mm && !value.y_mm && !value.yaw_deg)
    return "Fixture in position";
  return `Fixture x ${signed(value.x_mm, 1)} mm · y ${signed(value.y_mm, 1)} mm · yaw ${signed(value.yaw_deg, 1)}°`;
}
// 현장 요청 카드 4개를 교대 1개의 주문 큐로 바꾼다. 카드의 순서가 실행 순서다.
// 편집기의 보드, 지그 오차와 공급 결함의 선택은 쓰지 않는다.
export function shiftRequest() {
  return {
    items: Object.entries(scenarios).map(([name, scenario]) => ({
      name,
      order: { kind: "board", parts: scenario.parts },
      fixture: scenario.fixture,
    })),
  };
}
// 실행에 넣은 공급 결함의 요약이다. 결함이 없으면 빈 문자열이다.
export function supplyText(faults) {
  const missing = faults?.missing.length || 0,
    wrong = faults?.wrong_color.length || 0;
  const items = [];
  if (missing)
    items.push(`${missing} ${missing === 1 ? "part" : "parts"} missing`);
  if (wrong)
    items.push(`${wrong} wrong-color ${wrong === 1 ? "part" : "parts"}`);
  return items.length ? `Supply: ${items.join(", ")}` : "";
}

function shape(type, rotation) {
  return options.parts
    .find((item) => item.type === type)
    .shapes.find((item) => item.rotation_deg === rotation).cells;
}
function cellsOf(item) {
  return shape(item.type, item.rotation_deg).map(([row, col]) => [
    item.row + row,
    item.col + col,
  ]);
}
function owners() {
  const map = new Map();
  parts.forEach((item, index) => {
    for (const [row, col] of cellsOf(item)) map.set(`${row},${col}`, index);
  });
  return map;
}
// 대칭 형태는 여러 회전이 같은 셀을 쓴다. 그중 가장 작은 회전을 주문에 쓴다.
function canonicalRotation(type, rotation) {
  const cells = JSON.stringify(shape(type, rotation));
  return options.parts
    .find((item) => item.type === type)
    .shapes.find((item) => JSON.stringify(item.cells) === cells).rotation_deg;
}
// 커서 아래에 부품의 첫 셀을 둔다. 부품이 보드 안에 남도록 위치를 당긴다.
function candidate(row, col) {
  const cells = shape(tool.type, tool.rotation_deg);
  const height = Math.max(...cells.map(([r]) => r)) + 1;
  const width = Math.max(...cells.map(([, c]) => c)) + 1;
  const clamp = (value, limit) => Math.min(Math.max(value, 0), limit);
  const item = {
    ...tool,
    rotation_deg: canonicalRotation(tool.type, tool.rotation_deg),
    row: clamp(row - cells[0][0], options.board.rows - height),
    col: clamp(col - cells[0][1], options.board.cols - width),
  };
  const taken = owners();
  const placed = cellsOf(item);
  return {
    item,
    cells: placed,
    fits:
      parts.length < options.max_parts &&
      placed.every(([r, c]) => !taken.has(`${r},${c}`)),
  };
}

function cellAt(row, col) {
  return $("editor").querySelector(`[data-row="${row}"][data-col="${col}"]`);
}
function clearGhost() {
  for (const cell of $("editor").querySelectorAll(".ghost, .ghost-bad")) {
    cell.classList.remove("ghost", "ghost-bad");
    cell.firstElementChild.className = "tile";
  }
}
function showGhost(row, col) {
  clearGhost();
  if (owners().has(`${row},${col}`)) return;
  const { cells, fits } = candidate(row, col);
  for (const [r, c] of cells) {
    const cell = cellAt(r, c);
    if (!cell || cell.dataset.part) continue;
    cell.classList.add(fits ? "ghost" : "ghost-bad");
    cell.firstElementChild.classList.add(tool.color);
  }
}
function hint(message) {
  $("editor-hint").textContent = message || HINT;
  $("editor-hint").classList.toggle("warning", Boolean(message));
}
function paint() {
  const taken = owners();
  const { rows, cols } = options.board;
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < cols; col++) {
      const cell = cellAt(row, col);
      const index = taken.get(`${row},${col}`);
      const tile = cell.firstElementChild;
      tile.className = "tile";
      cell.classList.remove("ghost", "ghost-bad");
      if (index === undefined) {
        delete cell.dataset.part;
        cell.setAttribute("aria-label", `Row ${row}, col ${col}, empty`);
        continue;
      }
      const item = parts[index];
      cell.dataset.part = String(index);
      tile.classList.add(item.color);
      // 같은 부품의 이웃 셀은 틈 없이 잇는다.
      for (const [name, dr, dc] of [
        ["join-top", -1, 0],
        ["join-bottom", 1, 0],
        ["join-left", 0, -1],
        ["join-right", 0, 1],
      ])
        if (taken.get(`${row + dr},${col + dc}`) === index)
          tile.classList.add(name);
      const name = options.parts.find((p) => p.type === item.type).name;
      cell.setAttribute(
        "aria-label",
        `Row ${row}, col ${col}, ${item.color} ${name}. Remove`,
      );
    }
}
function act(row, col) {
  const index = owners().get(`${row},${col}`);
  if (index !== undefined) {
    parts = parts.filter((_, i) => i !== index);
    changed();
    return;
  }
  const { item, fits } = candidate(row, col);
  if (!fits) {
    hint(
      parts.length >= options.max_parts
        ? `The board has the maximum of ${options.max_parts} parts.`
        : "The part does not fit here. Rotate it or remove a part.",
    );
    return;
  }
  parts = [...parts, item];
  changed();
}
function buildEditor() {
  const editor = $("editor");
  const { rows, cols } = options.board;
  editor.style.setProperty("--cols", cols);
  editor.append(node("span"));
  for (let col = 0; col < cols; col++)
    editor.append(node("span", String(col), "axis"));
  for (let row = 0; row < rows; row++) {
    editor.append(node("span", String(row), "axis"));
    for (let col = 0; col < cols; col++) {
      const cell = node("button", undefined, "cell");
      cell.type = "button";
      cell.dataset.row = row;
      cell.dataset.col = col;
      cell.append(node("i", undefined, "tile"));
      cell.addEventListener("pointerenter", () => showGhost(row, col));
      cell.addEventListener("focus", () => showGhost(row, col));
      cell.addEventListener("blur", clearGhost);
      cell.addEventListener("click", () => act(row, col));
      editor.append(cell);
    }
  }
  editor.addEventListener("pointerleave", clearGhost);
}

function shapeIcon(cells) {
  const icon = node("span", undefined, "shape-icon");
  const filled = new Set(cells.map(([row, col]) => `${row},${col}`));
  for (let row = 0; row < 3; row++)
    for (let col = 0; col < 3; col++)
      icon.append(node("i", undefined, filled.has(`${row},${col}`) ? "on" : ""));
  return icon;
}
function choice(container, { label, pressed, onPick, lead }) {
  const button = node("button", undefined, "tool");
  button.type = "button";
  button.setAttribute("aria-pressed", String(pressed));
  button.append(lead, node("span", label));
  button.addEventListener("click", onPick);
  container.append(button);
}
function renderTools() {
  $("part-tools").replaceChildren();
  for (const item of options.parts)
    choice($("part-tools"), {
      label: item.name.replace(" tile", ""),
      pressed: item.type === tool.type,
      lead: shapeIcon(shape(item.type, tool.rotation_deg)),
      onPick: () => {
        tool = { ...tool, type: item.type };
        renderTools();
      },
    });
  $("color-tools").replaceChildren();
  for (const color of options.colors)
    choice($("color-tools"), {
      label: color,
      pressed: color === tool.color,
      lead: node("i", undefined, `swatch ${color}`),
      onPick: () => {
        tool = { ...tool, color };
        renderTools();
      },
    });
  $("rotation-value").textContent = `${tool.rotation_deg}°`;
}
function rotate() {
  tool = {
    ...tool,
    rotation_deg: (tool.rotation_deg + QUARTER_TURN) % FULL_TURN,
  };
  renderTools();
  clearGhost();
}

function sameParts(a, b) {
  return (
    a.length === b.length &&
    a.every((item, i) =>
      ["type", "color", "rotation_deg", "row", "col"].every(
        (key) => item[key] === b[i][key],
      ),
    )
  );
}
function kitTotal() {
  return kit.items.reduce((sum, item) => sum + item.count, 0);
}
function currentOrder() {
  if (orderMode === "kit") {
    const items = kit.items.filter((item) => item.count > 0);
    return items.length
      ? { kind: "kit", items, arrangement: kit.arrangement }
      : null;
  }
  return parts.length ? { kind: "board", parts } : null;
}
function countText(count, limit) {
  const minutes = Math.max(
    1,
    Math.round(
      (options.seconds_base + options.seconds_per_part * count) / 60,
    ),
  );
  return `${count} of ${limit} parts${count ? ` · about ${minutes} min` : ""}`;
}
function renderKit() {
  const full = kitTotal() >= options.kit.max_parts;
  kit.items.forEach((item, index) => {
    const control = kitControls[index];
    control.count.textContent = String(item.count);
    control.less.disabled = item.count === 0;
    control.more.disabled =
      full || item.count >= options.kit.parts[index].max_count;
  });
  for (const button of $("kit-arrangements").children)
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.value === kit.arrangement),
    );
  $("kit-count").textContent = countText(kitTotal(), options.kit.max_parts);
}
function kitChanged() {
  renderKit();
  schedulePreview();
}
function buildKit() {
  kit = {
    items: options.kit.parts.map(({ type }) => ({
      type,
      count: KIT_START[type] ?? 0,
    })),
    arrangement: options.kit.arrangements[0].value,
  };
  kitControls = kit.items.map((item, index) => {
    // 키트의 색은 서버가 종류마다 정한다. 화면은 색을 고르지 않는다.
    const { color } = options.kit.parts[index];
    const part = options.parts.find((option) => option.type === item.type);
    const label = `${color} ${part.name}`;
    const row = node("div", undefined, "kit-item");
    const name = node("span", undefined, "kit-part");
    name.append(
      node("i", undefined, `swatch ${color}`),
      shapeIcon(shape(item.type, 0)),
      node("span", label.replace(" tile", "")),
    );
    const step = (action, text, delta) => {
      const button = node("button", text);
      button.type = "button";
      button.setAttribute("aria-label", `${action} ${label}`);
      button.addEventListener("click", () => {
        item.count += delta;
        kitChanged();
      });
      return button;
    };
    const less = step("Remove one", "−", -1);
    const more = step("Add one", "+", 1);
    const count = node("output");
    const stepper = node("span", undefined, "stepper");
    stepper.append(less, count, more);
    row.append(name, stepper);
    $("kit-items").append(row);
    return { count, less, more };
  });
  for (const arrangement of options.kit.arrangements) {
    const button = node("button", arrangement.name, "chip-button");
    button.type = "button";
    button.dataset.value = arrangement.value;
    button.addEventListener("click", () => {
      kit.arrangement = arrangement.value;
      kitChanged();
    });
    $("kit-arrangements").append(button);
  }
  renderKit();
}

// 접수 결과를 화면 상태로 바꾼다. 수락한 주문만 실행 요청에 쓴다.
function readingOf(view) {
  const model = view.model.split("/").pop();
  if (view.outcome === "accepted")
    return {
      state: "ready",
      text: view.preview.instruction,
      note: `${view.order.kind === "kit" ? "Kit list" : "Board order"} · ${view.preview.part_count} parts · Read by ${model}`,
      order: view.order,
      intake_id: view.intake_id,
      supply: view.preview.supply_faults,
    };
  const text =
    view.outcome === "rejected"
      ? REJECT_TEXT[view.reject_reason]
      : view.outcome === "order_not_feasible"
        ? view.preview.issues[0]?.message || "The order is not feasible."
        : view.outcome === "order_invalid"
          ? "The model gave an order outside the order limits."
          : "The model did not give a valid answer. Read the order again.";
  return {
    state: "blocked",
    text,
    note: `No order · ${view.reject_reason || view.outcome} · ${model}`,
  };
}
async function readOrder() {
  const current = ++readTicket;
  reading = { state: "pending", text: "Reading the order…", note: "" };
  renderInstruction();
  let next;
  try {
    next = readingOf(
      await api("/api/orders/intake", {
        method: "POST",
        body: JSON.stringify({ text: $("goal").value }),
      }),
    );
  } catch (error) {
    next = {
      state: "blocked",
      text: `The order is not available. ${error.message}`,
      note: "",
    };
  }
  // 읽는 동안 문장을 고쳤으면 늦은 응답을 버린다.
  if (current !== readTicket) return;
  reading = next;
  renderInstruction();
}
// 서버가 지금의 주문에 정한 공급 결함이다. 주문을 읽기 전과 원문 전송에는 없다.
function supplyChoices() {
  const shown =
    mode !== "text" ? preview : $("send-raw").checked ? null : reading;
  return shown?.state === "ready" ? shown.supply : null;
}
function selectedFaults() {
  const faults = supply === "none" ? null : supplyChoices()?.[supply];
  return faults ? { faults } : {};
}
function renderSupply() {
  const raw = mode === "text" && $("send-raw").checked;
  const choices = supplyChoices();
  // 서버가 이 주문에 줄 수 없다고 한 결함은 고른 상태로 두지 않는다.
  if (choices && supply !== "none" && !choices[supply]) supply = "none";
  // 원문 전송은 결함을 보내지 않는다. 고른 결함은 남겨 두고 결함 없음을 표시한다.
  const shown = raw ? "none" : supply;
  for (const button of document.querySelectorAll("[data-supply]")) {
    const kind = button.dataset.supply;
    button.setAttribute("aria-pressed", String(kind === shown));
    button.disabled =
      kind !== "none" && (raw || Boolean(choices && !choices[kind]));
  }
  const color = choices?.wrong_color?.wrong_color[0].color;
  $("supply-hint").textContent = raw
    ? "A supply fault needs an order. The cell does not read an order from this text."
    : supply === "missing"
      ? `The feeder does not supply one part. It is the first part of the placement sequence.${SUPPLY_HIDDEN}`
      : supply === "wrong_color"
        ? `The feeder supplies one part in ${color || "a different color"}. It is the first part of the placement sequence.${SUPPLY_HIDDEN}`
        : choices && !choices.wrong_color
          ? "The feeder supplies each part of the plan. A wrong-color part is not available for this order."
          : "The feeder supplies each part of the plan.";
}
function renderInstruction() {
  const board = mode === "board";
  const raw = $("send-raw").checked;
  const shown = mode === "text" ? reading : preview;
  $("instruction-box").hidden = mode === "text" && raw;
  $("instruction").textContent = shown.text;
  $("instruction").className = `instruction ${shown.state}`;
  $("copy-instruction").disabled = preview.state !== "ready";
  $("read-order").disabled =
    raw || reading.state === "pending" || !$("goal").value.trim();
  $("reading-note").textContent = raw ? "" : reading.note;
  $("editor-count").textContent = options
    ? countText(parts.length, options.max_parts)
    : "";
  $("undo").disabled = $("clear").disabled = !parts.length;
  for (const button of document.querySelectorAll(".scenario")) {
    const scenario = scenarios[button.dataset.scenario];
    button.classList.toggle(
      "active",
      board &&
        sameParts(scenario.parts, parts) &&
        Object.keys(NOMINAL).every(
          (key) => scenario.fixture[key] === fixture[key],
        ),
    );
  }
  renderSupply();
  notify();
}
function schedulePreview() {
  clearTimeout(timer);
  const current = ++ticket;
  const order = currentOrder();
  if (!order) {
    preview = {
      state: "blocked",
      text:
        orderMode === "kit"
          ? "Add at least one part."
          : "Place at least one part.",
    };
    renderInstruction();
    return;
  }
  preview = { state: "pending", text: "Updating the instruction…" };
  renderInstruction();
  timer = setTimeout(async () => {
    let next;
    try {
      const result = await api("/api/orders/preview", {
        method: "POST",
        body: JSON.stringify(order),
      });
      next = result.instruction
        ? {
            state: "ready",
            text: result.instruction,
            supply: result.supply_faults,
          }
        : {
            state: "blocked",
            text: result.issues[0]?.message || "The order is not feasible.",
          };
    } catch (error) {
      next = {
        state: "blocked",
        text: `The instruction is not available. ${error.message}`,
      };
    }
    // 더 새로운 편집이 있으면 늦은 응답을 버린다.
    if (current !== ticket) return;
    preview = next;
    renderInstruction();
  }, PREVIEW_DELAY_MS);
}
function changed() {
  hint(null);
  paint();
  schedulePreview();
}

function renderFixture() {
  for (const [key, input] of Object.entries(fixtureInputs)) {
    $(input.id).value = fixture[key];
    $(`${input.id}-value`).textContent =
      `${signed(fixture[key], 1)}${input.unit}`;
  }
  const nominal = Object.keys(NOMINAL).every((key) => fixture[key] === 0);
  const shifted = Object.keys(SHIFTED).every(
    (key) => fixture[key] === SHIFTED[key],
  );
  $("fixture-nominal").setAttribute("aria-pressed", String(nominal));
  $("fixture-shifted").setAttribute("aria-pressed", String(shifted));
  // 보드 격자와 같은 방향이다. x는 row 방향(아래), y는 col 방향(오른쪽)이다.
  // yaw는 위에서 보아 반시계다. SVG의 rotate는 시계 방향이 양수다.
  const scale =
    (DIAGRAM_BOARD_PX * DIAGRAM_GAIN) /
    (options.board.cols * options.board.cell_size_mm);
  $("fixture-actual").setAttribute(
    "transform",
    `translate(${fixture.y_mm * scale} ${fixture.x_mm * scale}) rotate(${-fixture.yaw_deg * DIAGRAM_GAIN})`,
  );
  renderInstruction();
}
function setFixture(value) {
  fixture = { ...value };
  renderFixture();
}
function buildFixture() {
  for (const [key, input] of Object.entries(fixtureInputs)) {
    const element = $(input.id);
    element.min = options.fixture[key].min;
    element.max = options.fixture[key].max;
    element.step = input.step;
    element.addEventListener("input", () =>
      setFixture({ ...fixture, [key]: Number(element.value) }),
    );
  }
  $("fixture-nominal").addEventListener("click", () => setFixture(NOMINAL));
  $("fixture-shifted").addEventListener("click", () => setFixture(SHIFTED));
}

function setMode(next) {
  mode = next;
  for (const name of MODES) {
    $(`tab-${name}`).classList.toggle("active", name === mode);
    $(`tab-${name}`).setAttribute("aria-selected", String(name === mode));
    $(`pane-${name}`).hidden = name !== mode;
  }
  if (mode === "text" || mode === orderMode) {
    renderInstruction();
    return;
  }
  orderMode = mode;
  schedulePreview();
}
function goalChanged() {
  $("character-count").textContent = `${$("goal").value.length} / 4000`;
  readTicket++;
  reading = UNREAD;
  renderInstruction();
}

// 화면의 입력 상태를 실행 요청으로 바꾼다. `onChange`는 실행 버튼 상태를 다시 그린다.
export async function initOrder(onChange) {
  notify = onChange;
  $("tab-text").addEventListener("click", () => setMode("text"));
  $("goal").addEventListener("input", goalChanged);
  $("read-order").addEventListener("click", readOrder);
  $("send-raw").addEventListener("change", renderInstruction);
  for (const button of document.querySelectorAll("[data-supply]"))
    button.addEventListener("click", () => {
      supply = button.dataset.supply;
      renderInstruction();
    });
  goalChanged();
  // 자유 문장은 읽은 주문을 보낸다. 원문 전송을 고르면 문장을 그대로 보낸다.
  const form = {
    ready: () =>
      mode !== "text"
        ? preview.state === "ready"
        : $("send-raw").checked
          ? $("goal").value.trim().length > 0
          : reading.state === "ready",
    body: () =>
      mode !== "text"
        ? { order: currentOrder(), fixture, ...selectedFaults() }
        : $("send-raw").checked
          ? { prompt: $("goal").value, fixture }
          : {
              order: reading.order,
              intake_id: reading.intake_id,
              fixture,
              ...selectedFaults(),
            },
  };
  try {
    options = await api("/api/orders/options");
  } catch (error) {
    // 선택지를 읽지 못하면 두 주문 탭과 오차 입력을 쓸 수 없다. 자유 입력은 남긴다.
    $("tab-board").disabled = $("tab-kit").disabled = true;
    $("fixture").hidden = true;
    setMode("text");
    return {
      ...form,
      error: `The order forms are not available. ${error.message}`,
    };
  }
  $("tab-board").addEventListener("click", () => setMode("board"));
  $("tab-kit").addEventListener("click", () => setMode("kit"));
  buildEditor();
  buildKit();
  buildFixture();
  renderTools();
  $("rotate").addEventListener("click", rotate);
  $("undo").addEventListener("click", () => {
    parts = parts.slice(0, -1);
    changed();
  });
  $("clear").addEventListener("click", () => {
    parts = [];
    changed();
  });
  $("copy-instruction").addEventListener("click", () => {
    $("goal").value = preview.text;
    goalChanged();
  });
  document.addEventListener("keydown", (event) => {
    const typing = ["TEXTAREA", "INPUT"].includes(event.target.tagName);
    if (event.key.toLowerCase() !== "r" || typing || mode !== "board") return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    rotate();
  });
  for (const button of document.querySelectorAll(".scenario"))
    button.addEventListener("click", () => {
      const scenario = scenarios[button.dataset.scenario];
      parts = scenario.parts.map((item) => ({ ...item }));
      fixture = { ...scenario.fixture };
      renderFixture();
      changed();
    });
  parts = SIX_PART_KIT.map((item) => ({ ...item }));
  renderFixture();
  changed();
  return form;
}
