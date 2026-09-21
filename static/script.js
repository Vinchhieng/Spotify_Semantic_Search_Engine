const form = document.getElementById("search-form");
const input = document.getElementById("query");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const examplesEl = document.getElementById("examples");

function setStatus(message, isError = false) {
  if (!message) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    return;
  }

  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.className = isError ? "status error" : "status";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function renderResults(results) {
  resultsEl.innerHTML = "";

  results.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = "track";

    // Convert similarity score from 0–1 into percentage.
    const pct = Math.round(
      Math.max(0, Math.min(1, r.score)) * 100
    );

    // Create lyrics link.
    const href =
      r.link && r.link.startsWith("http")
        ? r.link
        : `https://www.lyrics.com${
            r.link ? "/" + r.link.replace(/^\//, "") : ""
          }`;

    li.innerHTML = `
      <span class="track-index">
        ${String(i + 1).padStart(2, "0")}
      </span>

      <div class="track-main">
        <p class="track-title">
          <a
            href="${href}"
            target="_blank"
            rel="noopener noreferrer"
          >
            ${escapeHtml(r.song)}
          </a>
        </p>

        <p class="track-artist">
          ${escapeHtml(r.artist)}
        </p>

        <p class="track-excerpt">
          “${escapeHtml(r.excerpt)}”
        </p>
      </div>

      <span class="track-score">
        ${pct}% match
      </span>
    `;

    resultsEl.appendChild(li);
  });
}

async function runSearch(query) {
  const q = query.trim();

  if (!q) return;

  resultsEl.innerHTML = "";

  setStatus("Finding songs that match this feeling…");

  try {
    const res = await fetch(
      `/api/search?q=${encodeURIComponent(q)}&top_k=12`
    );

    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }

    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      setStatus(
        "No matches found. Try describing the mood in another way."
      );
      return;
    }

    setStatus(
      `${data.count} semantic matches for “${data.query}”`
    );

    renderResults(data.results);

    document
      .querySelector(".results-wrap")
      ?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });

  } catch (err) {
    console.error(err);

    setStatus(
      "The search engine could not be reached. Check that FastAPI is running.",
      true
    );
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();

  runSearch(input.value);
});

examplesEl.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");

  if (!chip) return;

  input.value = chip.textContent.trim();

  runSearch(input.value);
});