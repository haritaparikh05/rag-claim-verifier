"use strict";

const VERDICTS = {
  supported: { label: "Supported", meaning: "The documentation confirms this claim." },
  contradicted: { label: "Contradicted", meaning: "The documentation states something that conflicts with this claim." },
  unverifiable: { label: "Unverifiable", meaning: "The documentation does not say, so this claim cannot be confirmed." },
};

const REQUEST_TIMEOUT_MS = 90000;
const MAX_BATCH_SIZE = 25;
const MAX_CLAIM_LENGTH = 500;

const $ = (id) => document.getElementById(id);

const el = {
  banner: $("banner"),
  product: $("product"),
  tabSingle: $("tab-single"),
  tabBatch: $("tab-batch"),
  panelSingle: $("panel-single"),
  panelBatch: $("panel-batch"),
  form: $("verify-form"),
  claim: $("claim"),
  examples: $("examples"),
  submit: $("submit"),
  spinner: $("spinner"),
  submitLabel: $("submit-label"),
  result: $("result"),
  badge: $("badge"),
  confidenceText: $("confidence-text"),
  meterFill: $("meter-fill"),
  latency: $("latency"),
  meaning: $("verdict-meaning"),
  claimEcho: $("claim-echo"),
  passage: $("passage"),
  explanation: $("explanation"),
  batchForm: $("batch-form"),
  batchClaims: $("batch-claims"),
  batchCount: $("batch-count"),
  batchExamples: $("batch-examples"),
  csvFile: $("csv-file"),
  batchSubmit: $("batch-submit"),
  batchStop: $("batch-stop"),
  batchProgress: $("batch-progress"),
  progressBar: $("progress-bar"),
  progressText: $("progress-text"),
  batchResults: $("batch-results"),
  batchRows: $("batch-rows"),
  csvDownload: $("csv-download"),
};

let products = [];

/* ---------- shared helpers ---------- */

function showBanner(message, kind = "error") {
  el.banner.textContent = message;
  el.banner.className = "banner " + kind;
  el.banner.hidden = false;
}

function hideBanner() {
  el.banner.hidden = true;
}

function errorText(data, status) {
  return data && typeof data.detail === "string" ? data.detail : "Something went wrong (error " + status + ").";
}

function selectedProduct() {
  return products.find((p) => p.product_id === el.product.value);
}

function formatLatency(ms) {
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : ms + " ms";
}

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- products ---------- */

async function loadProducts() {
  try {
    const response = await fetch("/products");
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      showBanner(errorText(data, response.status));
      return;
    }
    products = data;
  } catch (err) {
    showBanner("Cannot reach the server. Is it running?");
    return;
  }

  el.product.replaceChildren();
  const groups = new Map();
  for (const product of products) {
    if (!groups.has(product.category)) {
      const group = document.createElement("optgroup");
      group.label = product.category;
      groups.set(product.category, group);
      el.product.appendChild(group);
    }
    const option = document.createElement("option");
    option.value = product.product_id;
    option.textContent = product.chunks > 0 ? product.name : product.name + " (no documents loaded)";
    option.disabled = product.chunks === 0;
    groups.get(product.category).appendChild(option);
  }

  const firstAvailable = products.find((p) => p.chunks > 0);
  if (!firstAvailable) {
    showBanner(
      "No documents have been loaded yet. Run docker compose up (or python -m ingestion.ingest), then reload this page.",
      "warning"
    );
    return;
  }
  el.product.value = firstAvailable.product_id;
  el.product.disabled = false;
  el.submit.disabled = false;
  el.batchSubmit.disabled = false;
  renderExamples();
}

function renderExamples() {
  el.examples.replaceChildren();
  const product = selectedProduct();
  if (!product) return;
  for (const text of product.examples) {
    const chip = makeElement("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => {
      el.claim.value = text;
      el.claim.focus();
    });
    el.examples.appendChild(chip);
  }
}

/* ---------- tabs ---------- */

function selectTab(name) {
  const batch = name === "batch";
  el.tabSingle.classList.toggle("selected", !batch);
  el.tabSingle.setAttribute("aria-selected", String(!batch));
  el.tabBatch.classList.toggle("selected", batch);
  el.tabBatch.setAttribute("aria-selected", String(batch));
  el.panelSingle.hidden = batch;
  el.panelBatch.hidden = !batch;
  el.result.hidden = batch || el.result.dataset.has !== "1";
  hideBanner();
}

/* ---------- single claim ---------- */

function setLoading(loading) {
  el.submit.disabled = loading;
  el.product.disabled = loading;
  el.claim.disabled = loading;
  el.spinner.hidden = !loading;
  el.submitLabel.textContent = loading ? "Checking..." : "Check claim";
  for (const chip of el.examples.querySelectorAll("button")) chip.disabled = loading;
}

function renderResult(data, claim, product) {
  const verdict = VERDICTS[data.verdict] ? data.verdict : "unverifiable";

  el.badge.textContent = VERDICTS[verdict].label;
  el.badge.className = "badge " + verdict;
  el.meaning.textContent = VERDICTS[verdict].meaning;
  el.claimEcho.textContent = "Claim: \"" + claim + "\" for " + product.name;

  const percent = Math.max(0, Math.min(100, Math.round(data.confidence * 100)));
  el.confidenceText.textContent = percent + "% confidence";
  el.meterFill.style.width = percent + "%";
  el.latency.textContent = formatLatency(data.latency_ms);

  el.passage.className = "passage " + verdict;
  el.passage.textContent = data.cited_passage || "No passage cited: the documentation does not address this claim.";
  el.explanation.textContent = data.explanation;

  el.result.dataset.has = "1";
  el.result.hidden = false;
}

async function verify(event) {
  event.preventDefault();
  hideBanner();

  const claim = el.claim.value.trim();
  const product = selectedProduct();
  if (!product) {
    showBanner("Please choose a product.", "warning");
    return;
  }
  if (!claim) {
    showBanner("Please enter a claim to check.", "warning");
    el.claim.focus();
    return;
  }

  setLoading(true);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch("/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_id: product.product_id, claim }),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      el.result.hidden = true;
      el.result.dataset.has = "0";
      showBanner(errorText(data, response.status));
      return;
    }
    renderResult(data, claim, product);
  } catch (err) {
    el.result.hidden = true;
    el.result.dataset.has = "0";
    showBanner(
      err.name === "AbortError"
        ? "This is taking too long. The AI service may be busy; please try again."
        : "Cannot reach the server. Is it running?"
    );
  } finally {
    clearTimeout(timer);
    setLoading(false);
  }
}

/* ---------- batch: input ---------- */

function claimsFromTextarea() {
  return el.batchClaims.value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function updateBatchCount() {
  const count = claimsFromTextarea().length;
  el.batchCount.textContent = count + (count === 1 ? " claim" : " claims") + " (max " + MAX_BATCH_SIZE + ")";
  el.batchCount.classList.toggle("over", count > MAX_BATCH_SIZE);
}

// Minimal CSV reader: quoted fields, doubled quotes, commas and newlines inside quotes.
function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell !== "" || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function claimsFromCsv(text) {
  const rows = parseCsv(text.replace(/^﻿/, ""));
  const claims = rows.map((r) => (r[0] || "").replace(/\s+/g, " ").trim()).filter(Boolean);
  if (claims.length && ["claim", "claims"].includes(claims[0].toLowerCase())) claims.shift();
  return claims;
}

async function loadCsvFile() {
  const file = el.csvFile.files[0];
  el.csvFile.value = "";
  if (!file) return;
  hideBanner();
  const claims = claimsFromCsv(await file.text());
  if (!claims.length) {
    showBanner("No claims were found in that file. Put one claim per row in the first column.", "warning");
    return;
  }
  el.batchClaims.value = claims.join("\n");
  updateBatchCount();
  if (claims.length > MAX_BATCH_SIZE) {
    showBanner("That file has " + claims.length + " claims. Please keep it to " + MAX_BATCH_SIZE + " or fewer.", "warning");
  }
}

/* ---------- batch: results ---------- */

let batchRows = [];
let batchResults = [];
let batchClaims = [];
let batchController = null;

function setBatchRunning(running) {
  el.batchSubmit.disabled = running;
  el.batchClaims.disabled = running;
  el.batchExamples.disabled = running;
  el.product.disabled = running;
  el.batchStop.hidden = !running;
}

function prepareRows(claims) {
  el.batchRows.replaceChildren();
  batchRows = claims.map((claim, index) => {
    const tr = document.createElement("tr");
    const verdictCell = makeElement("td");
    verdictCell.appendChild(makeElement("span", "badge small pending", "Waiting"));
    const cells = {
      verdict: verdictCell,
      confidence: makeElement("td", "time", ""),
      passage: makeElement("td", "passage-cell", ""),
      time: makeElement("td", "time", ""),
    };
    tr.append(makeElement("td", "num", String(index + 1)), makeElement("td", "", claim), cells.verdict, cells.confidence, cells.passage, cells.time);
    el.batchRows.appendChild(tr);
    return cells;
  });
  batchResults = new Array(claims.length).fill(null);
}

function fillRow(item) {
  const cells = batchRows[item.index];
  if (!cells) return;
  batchResults[item.index] = item;
  cells.verdict.replaceChildren();
  if (item.error) {
    cells.verdict.appendChild(makeElement("span", "badge small failed", "Not checked"));
    cells.passage.textContent = item.error;
    cells.passage.classList.add("error-text");
    return;
  }
  const verdict = VERDICTS[item.verdict] ? item.verdict : "unverifiable";
  cells.verdict.appendChild(makeElement("span", "badge small " + verdict, VERDICTS[verdict].label));
  cells.confidence.textContent = Math.round(item.confidence * 100) + "%";
  cells.passage.textContent = item.cited_passage || "No passage cited";
  cells.time.textContent = formatLatency(item.latency_ms);
}

function markUnfinishedRows(label) {
  batchRows.forEach((cells, index) => {
    if (batchResults[index]) return;
    cells.verdict.replaceChildren(makeElement("span", "badge small failed", label));
  });
}

function handleBatchMessage(message, state) {
  if (message.type === "start") {
    state.total = message.total;
    el.progressBar.max = message.total;
    el.progressText.textContent = "Starting...";
  } else if (message.type === "result") {
    fillRow(message);
    state.finished += 1;
    el.progressBar.value = state.finished;
    el.progressText.textContent = "Checked " + state.finished + " of " + state.total;
  } else if (message.type === "done") {
    state.done = true;
    const failedText = message.failed ? ", " + message.failed + " not checked" : "";
    el.progressText.textContent =
      "Done: " + message.succeeded + " checked" + failedText + " in " + formatLatency(message.duration_ms);
  }
}

async function runBatch(event) {
  event.preventDefault();
  hideBanner();

  const product = selectedProduct();
  const claims = claimsFromTextarea();
  if (!product) {
    showBanner("Please choose a product.", "warning");
    return;
  }
  if (!claims.length) {
    showBanner("Please add at least one claim.", "warning");
    el.batchClaims.focus();
    return;
  }
  if (claims.length > MAX_BATCH_SIZE) {
    showBanner("You can check up to " + MAX_BATCH_SIZE + " claims at a time (you have " + claims.length + ").", "warning");
    return;
  }
  if (claims.some((claim) => claim.length > MAX_CLAIM_LENGTH)) {
    showBanner("Each claim must be " + MAX_CLAIM_LENGTH + " characters or fewer.", "warning");
    return;
  }

  batchClaims = claims;
  prepareRows(claims);
  el.batchResults.hidden = false;
  el.csvDownload.hidden = true;
  el.batchProgress.hidden = false;
  el.progressBar.max = claims.length;
  el.progressBar.value = 0;
  el.progressText.textContent = "Starting...";
  setBatchRunning(true);

  const state = { total: claims.length, finished: 0, done: false };
  batchController = new AbortController();
  try {
    const response = await fetch("/verify/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_id: product.product_id, claims }),
      signal: batchController.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      el.batchResults.hidden = true;
      el.batchProgress.hidden = true;
      showBanner(errorText(data, response.status));
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) handleBatchMessage(JSON.parse(line), state);
      }
    }
    if (!state.done) {
      markUnfinishedRows("Not finished");
      showBanner("The connection ended before every claim was checked. Results so far are shown below.", "warning");
    }
  } catch (err) {
    markUnfinishedRows("Stopped");
    if (err.name === "AbortError") {
      el.progressText.textContent = "Stopped after " + state.finished + " of " + state.total;
      showBanner("Stopped. The claims already checked are shown below.", "info");
    } else {
      showBanner("Cannot reach the server. The claims already checked are shown below.");
    }
  } finally {
    batchController = null;
    setBatchRunning(false);
    el.csvDownload.hidden = !batchResults.some(Boolean);
  }
}

/* ---------- batch: CSV download ---------- */

function csvCell(value) {
  let text = value === null || value === undefined ? "" : String(value);
  // A cell that starts with = + - @ can be run as a formula by spreadsheet programs; mark it as plain text.
  if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
  return '"' + text.replace(/"/g, '""') + '"';
}

function downloadCsv() {
  const header = ["claim", "verdict", "confidence", "cited_passage", "explanation", "latency_ms", "error"];
  const lines = [header.map(csvCell).join(",")];
  batchResults.forEach((item, index) => {
    const row = item
      ? [item.claim, item.verdict, item.confidence, item.cited_passage, item.explanation, item.latency_ms, item.error]
      : [batchClaims[index], "", "", "", "", "", "Not checked"];
    lines.push(row.map(csvCell).join(","));
  });
  const blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "claim-verifier-results.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

/* ---------- wiring ---------- */

el.product.addEventListener("change", renderExamples);
el.tabSingle.addEventListener("click", () => selectTab("single"));
el.tabBatch.addEventListener("click", () => selectTab("batch"));
el.form.addEventListener("submit", verify);
el.batchForm.addEventListener("submit", runBatch);
el.batchClaims.addEventListener("input", updateBatchCount);
el.batchExamples.addEventListener("click", () => {
  const product = selectedProduct();
  if (!product) return;
  el.batchClaims.value = product.examples.join("\n");
  updateBatchCount();
});
el.csvFile.addEventListener("change", loadCsvFile);
el.batchStop.addEventListener("click", () => batchController && batchController.abort());
el.csvDownload.addEventListener("click", downloadCsv);

updateBatchCount();
loadProducts();
