const emailInput = document.getElementById('emailInput');
const charCount = document.getElementById('charCount');
const scanBtn = document.getElementById('scanBtn');
const resultsSection = document.getElementById('resultsSection');
const errorBox = document.getElementById('errorBox');

const gaugeFill = document.getElementById('gaugeFill');
const gaugePct = document.getElementById('gaugePct');
const riskBadge = document.getElementById('riskBadge');
const verdictText = document.getElementById('verdictText');
const modelBreakdown = document.getElementById('modelBreakdown');
const modelCountEl = document.getElementById('modelCount');

const cleanedPreview = document.getElementById('cleanedPreview');

const MODEL_TAGS = {
  nb: 'Bag of Words · Malitha', cnn: 'Word Embeddings · Malitha',
  svm: 'TF-IDF · Shakkya', bilstm: 'Attention · Shakkya',
  logreg: 'TF-IDF · Nishen', gru: 'Sequence · Nishen',
};

async function refreshModelCount() {
  try {
    const res = await fetch('/models');
    const data = await res.json();
    modelCountEl.textContent = Object.keys(data).length;
  } catch (e) { /* non-fatal */ }
}
refreshModelCount();

const GAUGE_CIRCUMFERENCE = 540.35;

emailInput.addEventListener('input', () => {
  charCount.textContent = `${emailInput.value.length} characters`;
});

const RISK_COLORS = {
  low: 'var(--safe)',
  medium: 'var(--warn)',
  high: 'var(--risk)',
};

const VERDICT_COPY = {
  low: 'Looks legitimate',
  medium: 'Suspicious — treat with caution',
  high: 'Likely phishing',
};

function setGauge(prob, risk) {
  const offset = GAUGE_CIRCUMFERENCE * (1 - prob);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = RISK_COLORS[risk];
  gaugePct.textContent = `${Math.round(prob * 100)}%`;
}

function renderModelCards(modelsObj) {
  modelBreakdown.innerHTML = '';
  Object.entries(modelsObj).forEach(([key, m]) => {
    const pct = Math.round(m.probability * 100);
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <div class="model-card-head">
        <span class="model-name">${m.label}</span>
        <span class="model-tag">${MODEL_TAGS[key] || m.owner}</span>
      </div>
      <div class="meter-track">
        <div class="meter-fill ${key}" style="width:0%"></div>
      </div>
      <div class="model-card-foot">
        <span class="model-pct">${pct}%</span>
        <span class="model-risk" style="color:${RISK_COLORS[m.risk]}">${m.risk.toUpperCase()}</span>
      </div>`;
    modelBreakdown.appendChild(card);
    requestAnimationFrame(() => {
      card.querySelector('.meter-fill').style.width = `${pct}%`;
    });
  });
}

async function runScan() {
  const text = emailInput.value.trim();
  errorBox.classList.add('hidden');

  if (!text) {
    errorBox.textContent = 'Paste an email into the box first.';
    errorBox.classList.remove('hidden');
    return;
  }

  scanBtn.disabled = true;
  scanBtn.querySelector('.scan-btn-label').textContent = 'Scanning...';

  // reset gauge before animating to new value
  gaugeFill.style.transition = 'none';
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email_text: text }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Something went wrong.');
    }

    resultsSection.classList.remove('hidden');

    // force reflow so the transition re-triggers after the reset above
    void gaugeFill.getBoundingClientRect();
    gaugeFill.style.transition = '';

    requestAnimationFrame(() => {
      setGauge(data.combined.probability, data.combined.risk);
      renderModelCards(data.models);
    });

    riskBadge.className = `risk-badge ${data.combined.risk}`;
    riskBadge.textContent = data.combined.risk.toUpperCase() + ' RISK';
    verdictText.textContent = VERDICT_COPY[data.combined.risk];
    verdictText.textContent += ` (${Object.keys(data.models).length}/6 models agreeing on average)`;

    cleanedPreview.textContent = data.cleaned_preview + '...';

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
    resultsSection.classList.add('hidden');
  } finally {
    scanBtn.disabled = false;
    scanBtn.querySelector('.scan-btn-label').textContent = 'Run Scan';
  }
}

scanBtn.addEventListener('click', runScan);

emailInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    runScan();
  }
});
