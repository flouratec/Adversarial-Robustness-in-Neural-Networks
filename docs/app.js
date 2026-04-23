function formatPercent(value) {
  return `${value.toFixed(2)}%`;
}

function getScenarioMessage(modelKey, attackKey, metrics) {
  const labelMap = {
    baseline: "The clean baseline",
    robust: "The adversarially trained model"
  };
  const attackMap = {
    clean: "under clean evaluation remains strong.",
    fgsm: "under FGSM shows how one-step perturbations change reliability.",
    pgd: "under PGD reveals the true worst-case local weakness."
  };
  const value = metrics[modelKey][attackKey];

  if (attackKey === "clean") {
    return `${labelMap[modelKey]} ${attackMap[attackKey]}`;
  }
  if (value < 5) {
    return `${labelMap[modelKey]} ${attackMap[attackKey]} The performance collapse is severe.`;
  }
  if (value < 35) {
    return `${labelMap[modelKey]} ${attackMap[attackKey]} The model is still highly fragile.`;
  }
  return `${labelMap[modelKey]} ${attackMap[attackKey]} The defense materially improves robustness.`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderDownloads(downloads) {
  const wrap = document.getElementById("download-links");
  wrap.innerHTML = "";
  downloads.forEach(item => {
    const a = document.createElement("a");
    a.href = item.href;
    a.textContent = item.label;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    wrap.appendChild(a);
  });
}

function renderFindings(findings) {
  const list = document.getElementById("key-findings");
  list.innerHTML = "";
  findings.forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  });
}

function renderGallery(figures) {
  const tabs = document.getElementById("gallery-tabs");
  const image = document.getElementById("gallery-image");
  const caption = document.getElementById("gallery-caption");

  function activateFigure(figId) {
    const fig = figures.find(item => item.id === figId) || figures[0];
    image.src = fig.src;
    image.alt = fig.title;
    caption.textContent = `${fig.title} - ${fig.caption}`;
    [...tabs.querySelectorAll("button")].forEach(btn => {
      btn.classList.toggle("active", btn.dataset.id === fig.id);
    });
  }

  tabs.innerHTML = "";
  figures.forEach((fig, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = fig.id;
    button.textContent = fig.title;
    if (index === 0) button.classList.add("active");
    button.addEventListener("click", () => activateFigure(fig.id));
    tabs.appendChild(button);
  });

  activateFigure(figures[0].id);
}

function renderScenario(metrics) {
  const modelSelect = document.getElementById("model-select");
  const attackSelect = document.getElementById("attack-select");
  const bar = document.getElementById("scenario-bar");

  function update() {
    const modelKey = modelSelect.value;
    const attackKey = attackSelect.value;
    const accuracy = metrics[modelKey][attackKey];
    const clean = metrics[modelKey].clean;
    const drop = clean - accuracy;

    setText("scenario-accuracy", formatPercent(accuracy));
    setText("scenario-drop", `${drop >= 0 ? "-" : "+"}${Math.abs(drop).toFixed(2)} pp`);
    setText("scenario-f1", metrics[modelKey].f1.toFixed(4));
    setText("scenario-message", getScenarioMessage(modelKey, attackKey, metrics));
    bar.style.width = `${Math.max(0, Math.min(100, accuracy))}%`;
  }

  modelSelect.addEventListener("change", update);
  attackSelect.addEventListener("change", update);
  update();
}

function renderChart(metrics) {
  const ctx = document.getElementById("comparisonChart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Clean", "FGSM", "PGD"],
      datasets: [
        {
          label: "Baseline",
          data: [metrics.baseline.clean, metrics.baseline.fgsm, metrics.baseline.pgd],
          backgroundColor: "rgba(96,165,250,0.75)",
          borderColor: "rgba(96,165,250,1)",
          borderWidth: 1,
          borderRadius: 8
        },
        {
          label: "Adversarial training",
          data: [metrics.robust.clean, metrics.robust.fgsm, metrics.robust.pgd],
          backgroundColor: "rgba(52,211,153,0.75)",
          borderColor: "rgba(52,211,153,1)",
          borderWidth: 1,
          borderRadius: 8
        }
      ]
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#e5eefc" }
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#cbd5e1" },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            color: "#cbd5e1",
            callback: value => `${value}%`
          },
          grid: { color: "rgba(148,163,184,0.18)" }
        }
      }
    }
  });
}

function init() {
  const data = window.DASHBOARD_DATA;
  if (!data) return;

  document.getElementById("project-title").textContent = data.project.title;
  document.getElementById("project-subtitle").textContent = data.project.subtitle;
  document.getElementById("project-note").textContent = data.project.note;
  document.getElementById("dataset-line").textContent = data.project.dataset;

  const pgdGain = data.metrics.robust.pgd - data.metrics.baseline.pgd;
  setText("kpi-clean-baseline", formatPercent(data.metrics.baseline.clean));
  setText("kpi-pgd-baseline", formatPercent(data.metrics.baseline.pgd));
  setText("kpi-pgd-robust", formatPercent(data.metrics.robust.pgd));
  setText("kpi-pgd-gain", `+${pgdGain.toFixed(2)} pp`);

  renderDownloads(data.downloads);
  renderFindings(data.keyFindings);
  renderGallery(data.figures);
  renderScenario(data.metrics);
  renderChart(data.metrics);
}

window.addEventListener("DOMContentLoaded", init);
