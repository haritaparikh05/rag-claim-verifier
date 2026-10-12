"use strict";

const content = document.getElementById("content");
const banner = document.getElementById("banner");

const VERDICT_LABELS = { supported: "Supported", contradicted: "Contradicted", unverifiable: "Unverifiable" };

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function percent(fraction) {
  return (fraction * 100).toFixed(1) + "%";
}

function verdictBadge(verdict) {
  const key = VERDICT_LABELS[verdict] ? verdict : "unverifiable";
  return make("span", "badge small " + key, VERDICT_LABELS[key]);
}

function statBlock(summary) {
  const block = make("div", "stat");
  block.appendChild(make("div", "stat-number", percent(summary.accuracy)));
  const detail = make("div", "stat-detail");
  detail.appendChild(make("div", "", summary.correct + " of " + summary.cases + " correct"));
  detail.appendChild(
    make("div", "muted", "95% range: " + percent(summary.interval[0]) + " to " + percent(summary.interval[1]))
  );
  block.appendChild(detail);
  return block;
}

function perClassTable(perClass) {
  const table = make("table", "compact");
  const head = make("tr");
  for (const label of ["Expected verdict", "Correct"]) head.appendChild(make("th", "", label));
  table.appendChild(make("thead")).appendChild(head);
  const body = make("tbody");
  for (const verdict of Object.keys(VERDICT_LABELS)) {
    const row = make("tr");
    const labelCell = make("td");
    labelCell.appendChild(verdictBadge(verdict));
    row.appendChild(labelCell);
    row.appendChild(make("td", "", perClass[verdict].correct + " of " + perClass[verdict].total));
    body.appendChild(row);
  }
  table.appendChild(body);
  return table;
}

function missesTable(failures) {
  if (!failures.length) return make("p", "muted", "No misses.");
  const wrapper = make("div", "table-scroll");
  const table = make("table");
  const head = make("tr");
  for (const label of ["Claim", "Product", "Expected", "Got"]) head.appendChild(make("th", "", label));
  table.appendChild(make("thead")).appendChild(head);
  const body = make("tbody");
  for (const miss of failures) {
    const row = make("tr");
    row.appendChild(make("td", "", miss.claim));
    row.appendChild(make("td", "num", miss.product_id || ""));
    const expected = make("td");
    expected.appendChild(verdictBadge(miss.expected));
    const got = make("td");
    got.appendChild(verdictBadge(miss.actual));
    row.append(expected, got);
    body.appendChild(row);
  }
  table.appendChild(body);
  wrapper.appendChild(table);
  return wrapper;
}

function productSetCard(set) {
  const card = make("section", "card");
  card.appendChild(make("h2", "section-title", set.title));
  card.appendChild(make("p", "muted", set.description));
  if (!set.available) {
    card.appendChild(make("p", "banner warning", "The result file " + set.file + " is not available."));
    return card;
  }
  card.appendChild(statBlock(set));
  card.appendChild(perClassTable(set.per_class));
  card.appendChild(make("h3", "subtitle", "Misses"));
  card.appendChild(missesTable(set.failures));
  card.appendChild(make("p", "source", "Source: data/eval/" + set.file));
  return card;
}

function feverCard(fever) {
  const card = make("section", "card");
  card.appendChild(make("h2", "section-title", fever.title));
  card.appendChild(make("p", "muted", fever.description));
  if (!fever.samples.length) {
    card.appendChild(make("p", "banner warning", "No FEVER result files are available."));
    return card;
  }
  const wrapper = make("div", "table-scroll");
  const table = make("table");
  const head = make("tr");
  for (const label of ["Sample (random seed)", "Correct", "Accuracy", "95% range"]) head.appendChild(make("th", "", label));
  table.appendChild(make("thead")).appendChild(head);
  const body = make("tbody");
  for (const sample of fever.samples) {
    const row = make("tr");
    row.appendChild(make("td", "", "Seed " + sample.seed));
    row.appendChild(make("td", "num", sample.correct + " of " + sample.cases));
    row.appendChild(make("td", "", percent(sample.accuracy)));
    row.appendChild(make("td", "num", percent(sample.interval[0]) + " to " + percent(sample.interval[1])));
    body.appendChild(row);
  }
  table.appendChild(body);
  wrapper.appendChild(table);
  card.appendChild(wrapper);
  card.appendChild(make("p", "", "Mean across the three samples: " + percent(fever.mean_accuracy)));
  card.appendChild(make("p", "source", "Source: data/eval/fever_eval_results*.json"));
  return card;
}

function notesCard(notes) {
  const card = make("section", "card");
  card.appendChild(make("h2", "section-title", "How to read these numbers"));
  const list = make("ul", "notes-list");
  for (const note of notes) list.appendChild(make("li", "", note));
  card.appendChild(list);
  return card;
}

async function load() {
  try {
    const response = await fetch("/api/results");
    if (!response.ok) throw new Error("status " + response.status);
    const data = await response.json();
    content.replaceChildren();
    for (const set of data.product_sets) content.appendChild(productSetCard(set));
    content.appendChild(feverCard(data.fever));
    content.appendChild(notesCard(data.notes));
  } catch (err) {
    content.replaceChildren();
    banner.textContent = "Could not load the results. Is the server running?";
    banner.className = "banner error";
    banner.hidden = false;
  }
}

load();
