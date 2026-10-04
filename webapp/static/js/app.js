/**
 * Marathi News Classifier - Client Application Logic (Phase 8.7)
 * Premium NLP Research Laboratory Experience
 * - Live Devanagari & Latin text analytics
 * - Dynamic SVG Donut Chart generation (Zero external libraries)
 * - High-contrast horizontal probability distributions
 * - Feature-level TF-IDF explainability showpiece
 * - Interactive pipeline inspection with parameter details
 * - Interactive category tab switching
 * - Timeline session history & UTF-8 BOM CSV export
 */

document.addEventListener("DOMContentLoaded", function () {
  // Elements - Main Classifier
  const headlineInput = document.getElementById("headline-input");
  const statWords = document.getElementById("stat-words");
  const statChars = document.getElementById("stat-chars");
  const statDevanagari = document.getElementById("stat-devanagari");
  const statLatin = document.getElementById("stat-latin");
  const statDigits = document.getElementById("stat-digits");

  const btnClassify = document.getElementById("btn-classify");
  const btnClear = document.getElementById("btn-clear");
  const btnRandomQuick = document.getElementById("btn-random-quick");
  const btnRandomLoad = document.getElementById("btn-random-load");
  const randomCategorySelect = document.getElementById("random-category-select");

  const loadingBox = document.getElementById("loading-box");
  const errorBanner = document.getElementById("error-banner");
  const resultSection = document.getElementById("result-section");

  const timelineContainer = document.getElementById("timeline-container");
  const btnClearHistory = document.getElementById("btn-clear-history");
  const btnDownloadCsv = document.getElementById("btn-download-csv");

  let latestPredictionData = null;

  // Semantic Category Color Palettes
  const categoryColorMap = {
    "Technology & Science": "#2563EB",
    "Politics": "#6366F1",
    "Business": "#059669",
    "Sports": "#F59E0B",
    "Entertainment": "#E11D48"
  };

  const categoryFillMap = {
    "Technology & Science": "fill-tech",
    "Politics": "fill-politics",
    "Business": "fill-business",
    "Sports": "fill-sports",
    "Entertainment": "fill-entertainment"
  };

  const categoryThemeMap = {
    "Technology & Science": "theme-tech",
    "Politics": "theme-politics",
    "Business": "theme-business",
    "Sports": "theme-sports",
    "Entertainment": "theme-entertainment"
  };

  // ==========================================================================
  // 1. Live Headline Text Diagnostics (Input Profile)
  // ==========================================================================
  function updateDiagnostics() {
    if (!headlineInput) return;
    const text = headlineInput.value || "";
    const trimmed = text.trim();

    const words = trimmed ? trimmed.split(/\s+/).length : 0;
    const chars = text.length;

    let devanagariCount = 0;
    let latinCount = 0;
    let digitCount = 0;

    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      // Devanagari block: U+0900 to U+097F
      if (code >= 0x0900 && code <= 0x097F) {
        devanagariCount++;
      } else if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) {
        latinCount++;
      }

      // ASCII digits (0-9) or Devanagari digits (०-९: 0x0966 - 0x096F)
      if ((code >= 48 && code <= 57) || (code >= 0x0966 && code <= 0x096F)) {
        digitCount++;
      }
    }

    const totalLetterChars = devanagariCount + latinCount;
    let devPercent = 0;
    let latinPercent = 0;

    if (totalLetterChars > 0) {
      devPercent = Math.round((devanagariCount / totalLetterChars) * 100);
      latinPercent = Math.round((latinCount / totalLetterChars) * 100);
    }

    if (statWords) statWords.textContent = words;
    if (statChars) statChars.textContent = chars;
    if (statDevanagari) statDevanagari.textContent = `${devPercent}%`;
    if (statLatin) statLatin.textContent = `${latinPercent}%`;
    if (statDigits) statDigits.textContent = digitCount;
  }

  if (headlineInput) {
    headlineInput.addEventListener("input", updateDiagnostics);
  }

  // ==========================================================================
  // 2. Quick Sample Chips & Clear
  // ==========================================================================
  const sampleChips = document.querySelectorAll(".sample-chip");
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", function () {
      const text = this.getAttribute("data-sample");
      if (text && headlineInput) {
        headlineInput.value = text;
        updateDiagnostics();
        hideError();
        headlineInput.focus();
      }
    });
  });

  if (btnClear) {
    btnClear.addEventListener("click", function () {
      if (headlineInput) {
        headlineInput.value = "";
        updateDiagnostics();
        hideError();
        headlineInput.focus();
      }
    });
  }

  // ==========================================================================
  // 3. Random Dataset Headline
  // ==========================================================================
  async function fetchRandomHeadline(category = "all") {
    try {
      showLoading(true, "Sampling verified headline from dataset...");
      hideError();

      const url = `/api/random-headline?category=${encodeURIComponent(category)}`;
      const resp = await fetch(url);
      const data = await resp.json();

      if (!resp.ok || !data.success) {
        throw new Error(data.error || "Failed to sample headline");
      }

      if (headlineInput) {
        headlineInput.value = data.headline;
        updateDiagnostics();
        headlineInput.focus();
      }
    } catch (err) {
      showError(err.message);
    } finally {
      showLoading(false);
    }
  }

  if (btnRandomLoad) {
    btnRandomLoad.addEventListener("click", function () {
      const cat = randomCategorySelect ? randomCategorySelect.value : "all";
      fetchRandomHeadline(cat);
    });
  }

  if (btnRandomQuick) {
    btnRandomQuick.addEventListener("click", function () {
      fetchRandomHeadline("all");
    });
  }

  // ==========================================================================
  // 4. Headline Classification Trigger
  // ==========================================================================
  if (btnClassify) {
    btnClassify.addEventListener("click", async function () {
      const headline = headlineInput ? headlineInput.value.trim() : "";

      if (!headline) {
        showError("Please enter a Marathi news headline.");
        if (headlineInput) headlineInput.focus();
        return;
      }

      try {
        showLoading(true, "Evaluating headline through feature pipeline...");
        hideError();
        if (resultSection) resultSection.style.display = "none";

        // Trigger sequential pipeline stage highlight
        animatePipelineStages();

        const resp = await fetch("/api/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ headline: headline }),
        });

        const data = await resp.json();

        if (!resp.ok || !data.success) {
          throw new Error(data.error || "Unable to classify the headline. Please try again.");
        }

        latestPredictionData = data;
        renderPredictionReveal(data);
        savePredictionToHistory(data);
      } catch (err) {
        showError(err.message);
      } finally {
        showLoading(false);
      }
    });
  }

  // ==========================================================================
  // 5. Prediction Reveal ("The Demo Moment")
  // ==========================================================================
  function renderPredictionReveal(data) {
    if (!resultSection) return;

    const resHeadline = document.getElementById("res-headline");
    const resCategoryName = document.getElementById("res-category-name");
    const resProbability = document.getElementById("res-probability");
    const predictionAccentLine = document.getElementById("prediction-accent-line");
    const distContainer = document.getElementById("dist-container");
    const top3HierarchyList = document.getElementById("top3-hierarchy-list");
    const explainTermsList = document.getElementById("explain-terms-list");
    const explainCategoryTag = document.getElementById("explain-category-tag");
    const explainDisclaimer = document.getElementById("explain-disclaimer");

    const catName = data.predicted_category;
    const catColor = categoryColorMap[catName] || "#2563EB";
    const fillClass = categoryFillMap[catName] || "fill-tech";
    const themeClass = categoryThemeMap[catName] || "theme-tech";

    // 1. Headline quote & Category Banner
    if (resHeadline) resHeadline.textContent = `"${data.headline}"`;
    if (resCategoryName) {
      resCategoryName.textContent = catName;
      resCategoryName.style.color = catColor;
    }
    if (predictionAccentLine) {
      predictionAccentLine.className = `prediction-accent-line ${fillClass}`;
    }

    if (resProbability) {
      const probPct = (data.model_probability * 100).toFixed(2);
      resProbability.textContent = `${probPct}%`;
    }

    // 2. Render SVG Donut Chart (Pure SVG)
    renderSvgDonutChart(data.all_probabilities, catName);

    // 3. Render High-Contrast Horizontal Probability Distribution
    if (distContainer && data.all_probabilities) {
      distContainer.innerHTML = "";
      const sortedClasses = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

      sortedClasses.forEach(([name, prob]) => {
        const cFill = categoryFillMap[name] || "fill-tech";
        const cColor = categoryColorMap[name] || "#2563EB";
        const pct = (prob * 100).toFixed(2);

        const row = document.createElement("div");
        row.className = "contrast-bar-row";
        row.innerHTML = `
          <div class="contrast-bar-meta">
            <span style="color:${cColor};">${name}</span>
            <span style="font-family:var(--font-mono);">${pct}%</span>
          </div>
          <div class="contrast-bar-track">
            <div class="contrast-bar-fill ${cFill}" style="width: 0%" data-target-width="${pct}%"></div>
          </div>
        `;
        distContainer.appendChild(row);
      });
    }

    // 4. Render Top-3 Predictions Hierarchy (#1 Dominates)
    if (top3HierarchyList && data.top_predictions) {
      top3HierarchyList.innerHTML = "";
      data.top_predictions.forEach((item, index) => {
        const isRank1 = index === 0;
        const itemColor = categoryColorMap[item.category] || "#2563EB";
        const pct = (item.probability * 100).toFixed(2);

        const card = document.createElement("div");
        card.className = `top3-hier-card ${isRank1 ? 'rank-1' : ''}`;
        card.innerHTML = `
          <div class="top3-hier-header">
            <span class="top3-rank-pill">#${index + 1} ${isRank1 ? 'Top Pick' : ''}</span>
            <span class="top3-hier-pct" style="color:${itemColor};">${pct}%</span>
          </div>
          <div class="top3-hier-cat">${item.category}</div>
        `;
        top3HierarchyList.appendChild(card);
      });
    }

    // 5. Render Explainability Showpiece (Why Did The Model Choose This?)
    if (explainTermsList && data.explainability) {
      explainTermsList.innerHTML = "";
      if (explainCategoryTag) {
        explainCategoryTag.className = `category-pill-lg ${themeClass}`;
        explainCategoryTag.textContent = catName;
      }
      if (explainDisclaimer && data.explainability.disclaimer) {
        explainDisclaimer.textContent = data.explainability.disclaimer;
      }

      const topFeatures = data.explainability.top_features || [];
      if (topFeatures.length === 0) {
        explainTermsList.innerHTML = `
          <div style="font-size: 0.88rem; color: var(--text-muted); font-style: italic; padding: 0.5rem 0;">
            No out-of-vocabulary terms available for direct linear decomposition in this input.
          </div>
        `;
      } else {
        const maxScore = Math.max(...topFeatures.map(f => Math.abs(f.contribution)), 0.001);

        topFeatures.forEach((feat) => {
          const widthPct = Math.min(100, Math.max(12, (Math.abs(feat.contribution) / maxScore) * 100)).toFixed(1);
          const scoreSign = feat.contribution >= 0 ? `+${feat.contribution.toFixed(2)}` : feat.contribution.toFixed(2);
          const barFill = feat.contribution >= 0 ? fillClass : "fill-entertainment";

          const row = document.createElement("div");
          row.className = "contrib-row";
          row.innerHTML = `
            <span class="contrib-term-name">${feat.term}</span>
            <div class="contrib-track-wrapper">
              <div class="contrib-track">
                <div class="contrib-fill ${barFill}" style="width: 0%" data-target-width="${widthPct}%"></div>
              </div>
            </div>
            <span class="contrib-value-badge">${scoreSign}</span>
          `;
          explainTermsList.appendChild(row);
        });
      }
    }

    // Reveal Result Section with Smooth Transition
    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // Animate fill widths
    setTimeout(() => {
      document.querySelectorAll(".contrast-bar-fill, .contrib-fill").forEach((bar) => {
        const target = bar.getAttribute("data-target-width");
        if (target) bar.style.width = target;
      });
    }, 60);
  }

  // ==========================================================================
  // 6. SVG Donut / Radial Chart Generator (Pure SVG - Zero External CDN)
  // ==========================================================================
  function renderSvgDonutChart(probabilities, topCategory) {
    const svgElem = document.getElementById("donut-svg");
    const centerPct = document.getElementById("donut-center-pct");
    const centerLabel = document.getElementById("donut-center-label");
    if (!svgElem || !probabilities) return;

    svgElem.innerHTML = "";

    const radius = 70;
    const cx = 100;
    const cy = 100;
    const strokeWidth = 22;
    const circumference = 2 * Math.PI * radius; // ~439.82

    // Background track ring
    const bgCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    bgCircle.setAttribute("cx", cx);
    bgCircle.setAttribute("cy", cy);
    bgCircle.setAttribute("r", radius);
    bgCircle.setAttribute("fill", "transparent");
    bgCircle.setAttribute("stroke", "#CBD5E1");
    bgCircle.setAttribute("stroke-width", strokeWidth);
    svgElem.appendChild(bgCircle);

    // Sort categories descending
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    let accumulatedOffset = 0;

    sorted.forEach(([cat, prob]) => {
      const arcLength = prob * circumference;
      const color = categoryColorMap[cat] || "#3155D9";

      const arc = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      arc.setAttribute("cx", cx);
      arc.setAttribute("cy", cy);
      arc.setAttribute("r", radius);
      arc.setAttribute("fill", "transparent");
      arc.setAttribute("stroke", color);
      arc.setAttribute("stroke-width", strokeWidth);
      arc.setAttribute("stroke-dasharray", `${arcLength} ${circumference}`);
      arc.setAttribute("stroke-dashoffset", -accumulatedOffset);
      arc.setAttribute("transform", `rotate(-90 ${cx} ${cy})`);
      arc.style.transition = "stroke-dasharray 0.8s ease, stroke-dashoffset 0.8s ease";

      svgElem.appendChild(arc);
      accumulatedOffset += arcLength;
    });

    if (centerPct && probabilities[topCategory]) {
      centerPct.textContent = `${(probabilities[topCategory] * 100).toFixed(1)}%`;
    }
    if (centerLabel) {
      centerLabel.textContent = topCategory;
    }
  }

  // ==========================================================================
  // 7. Sequential Pipeline Animation
  // ==========================================================================
  function animatePipelineStages() {
    const stages = [
      "stage-input",
      "stage-preprocess",
      "stage-tfidf",
      "stage-classifier",
      "stage-prediction"
    ];

    stages.forEach((id, index) => {
      const elem = document.getElementById(id);
      if (!elem) return;
      elem.classList.remove("active-stage");
      setTimeout(() => {
        elem.classList.add("active-stage");
        setTimeout(() => {
          elem.classList.remove("active-stage");
        }, 500);
      }, index * 130);
    });
  }

  // ==========================================================================
  // 8. Interactive Pipeline Inspection (Click-to-inspect Details)
  // ==========================================================================
  const stageDetailsMap = {
    "input": {
      title: "Stage 01: Raw Marathi Headline String",
      body: "Accepts raw Unicode Marathi text containing Devanagari script, punctuation, digits, and legitimate English/Latin loanwords (e.g. 5G, ISRO, smartphone, GST)."
    },
    "preprocess": {
      title: "Stage 02: Preprocessing & Normalization (src/preprocessing.py)",
      body: "Applies Unicode NFC normalization, smart quote and dash standardization, and Latin lowercasing while strictly preserving Marathi Devanagari matras, numbers, and suffixes without stop-word deletion."
    },
    "tfidf": {
      title: "Stage 03: Sublinear TF-IDF Feature Extraction (models/tfidf_vectorizer.joblib)",
      body: "Transforms normalized headline into a 29,455-dimensional sparse numerical vector using custom Devanagari regex token_pattern=r'[\\u0900-\\u097F\\w]+', unigram+bigram (1, 2) n-grams, min_df=2, and sublinear term frequency scaling."
    },
    "classifier": {
      title: "Stage 04: Balanced Logistic Regression (models/final_classifier.joblib)",
      body: "Evaluates the 29,455 feature dimensions with inverse class frequency weights (class_weight='balanced') to counteract distribution skew, computing logits for each of the 5 categories."
    },
    "prediction": {
      title: "Stage 05: Softmax Probability Distribution & Prediction Output",
      body: "Computes softmax probability distribution across discrete classes (Technology & Science, Politics, Business, Sports, Entertainment) and outputs the predicted category, ranked top-3 classes, and term-level linear feature contributions."
    }
  };

  const pipelineCards = document.querySelectorAll(".pipeline-node-card");
  const pipelineDetailCard = document.getElementById("pipeline-detail-card");
  const pipelineDetailTitle = document.getElementById("pipeline-detail-title");
  const pipelineDetailBody = document.getElementById("pipeline-detail-body");

  pipelineCards.forEach((card) => {
    card.addEventListener("click", function () {
      const stageKey = this.getAttribute("data-stage");
      const details = stageDetailsMap[stageKey];
      if (details && pipelineDetailCard && pipelineDetailTitle && pipelineDetailBody) {
        pipelineCards.forEach(c => c.classList.remove("active-stage"));
        this.classList.add("active-stage");
        pipelineDetailTitle.textContent = details.title;
        pipelineDetailBody.textContent = details.body;
        pipelineDetailCard.style.display = "block";
      }
    });
  });

  // ==========================================================================
  // 9. Session Prediction Timeline & Storage
  // ==========================================================================
  const STORAGE_KEY = "marathi_news_classifier_history_v2";

  function getHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function savePredictionToHistory(data) {
    try {
      const history = getHistory();
      const newEntry = {
        headline: data.headline,
        category: data.predicted_category,
        probability: `${(data.model_probability * 100).toFixed(2)}%`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };
      history.unshift(newEntry);
      if (history.length > 20) history.pop();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
      renderTimeline();
    } catch (e) {
      console.warn("Could not save history:", e);
    }
  }

  function renderTimeline() {
    if (!timelineContainer) return;
    const history = getHistory();

    if (history.length === 0) {
      timelineContainer.innerHTML = `
        <div style="background:#FFFFFF; border:1px dashed var(--border-light); border-radius:var(--radius-lg); padding:2rem; text-align:center; color:var(--text-muted); font-size:0.9rem;">
          No predictions in current session history. Enter or select a headline above to run inference.
        </div>
      `;
      return;
    }

    timelineContainer.innerHTML = "";
    history.forEach((item) => {
      const themeClass = categoryThemeMap[item.category] || "theme-tech";
      const card = document.createElement("div");
      card.className = "timeline-entry-card";
      card.innerHTML = `
        <div class="timeline-entry-main">
          <div class="timeline-entry-meta">
            <span class="timeline-entry-category ${themeClass}">${item.category}</span>
            <span class="timeline-entry-time">${item.timestamp}</span>
          </div>
          <div class="timeline-entry-text">${item.headline}</div>
        </div>
        <div class="timeline-entry-score">
          <div class="timeline-entry-score-num">${item.probability}</div>
          <div class="timeline-entry-score-tag">Model Prob</div>
        </div>
      `;
      timelineContainer.appendChild(card);
    });
  }

  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", function () {
      localStorage.removeItem(STORAGE_KEY);
      renderTimeline();
    });
  }

  // ==========================================================================
  // 10. Download Prediction Result CSV
  // ==========================================================================
  if (btnDownloadCsv) {
    btnDownloadCsv.addEventListener("click", function () {
      if (!latestPredictionData) return;

      const top1 = latestPredictionData.top_predictions[0] ? `${latestPredictionData.top_predictions[0].category} (${(latestPredictionData.top_predictions[0].probability * 100).toFixed(2)}%)` : "";
      const top2 = latestPredictionData.top_predictions[1] ? `${latestPredictionData.top_predictions[1].category} (${(latestPredictionData.top_predictions[1].probability * 100).toFixed(2)}%)` : "";
      const top3 = latestPredictionData.top_predictions[2] ? `${latestPredictionData.top_predictions[2].category} (${(latestPredictionData.top_predictions[2].probability * 100).toFixed(2)}%)` : "";

      const headers = ["Headline", "Predicted Category", "Model Probability", "Top 1", "Top 2", "Top 3", "Timestamp"];
      const row = [
        `"${latestPredictionData.headline.replace(/"/g, '""')}"`,
        `"${latestPredictionData.predicted_category}"`,
        `"${(latestPredictionData.model_probability * 100).toFixed(2)}%"`,
        `"${top1}"`,
        `"${top2}"`,
        `"${top3}"`,
        `"${new Date().toISOString()}"`
      ];

      const csvContent = "\uFEFF" + headers.join(",") + "\n" + row.join(",");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `prediction_report_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // ==========================================================================
  // 11. Interactive Category Tabs on Dataset Page
  // ==========================================================================
  const datasetTabButtons = document.querySelectorAll(".dataset-tab-btn");
  const explorerTitle = document.getElementById("explorer-title");
  const explorerCount = document.getElementById("explorer-count");
  const explorerPct = document.getElementById("explorer-pct");
  const explorerLen = document.getElementById("explorer-len");
  const explorerShareBar = document.getElementById("explorer-share-bar");
  const explorerExamplesList = document.getElementById("explorer-examples-list");

  async function loadCategoryTabDetails(categoryName) {
    if (!explorerExamplesList) return;
    try {
      const resp = await fetch(`/api/category/${encodeURIComponent(categoryName)}`);
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || "Failed to load category details");
      }

      if (explorerTitle) explorerTitle.textContent = data.category;
      if (explorerCount) explorerCount.textContent = Number(data.record_count).toLocaleString();
      if (explorerPct) explorerPct.textContent = `${data.percentage}%`;
      if (explorerLen) explorerLen.textContent = `${data.avg_headline_length || '11.8'} words`;

      if (explorerShareBar) {
        const fillClass = categoryFillMap[data.category] || "fill-tech";
        explorerShareBar.className = `contrast-bar-fill ${fillClass}`;
        explorerShareBar.style.width = `${data.percentage}%`;
      }

      explorerExamplesList.innerHTML = "";
      data.examples.forEach((ex, idx) => {
        const item = document.createElement("div");
        item.style.background = "#FFFFFF";
        item.style.border = "1px solid var(--border-light)";
        item.style.borderRadius = "var(--radius-md)";
        item.style.padding = "0.85rem 1.15rem";
        item.style.fontSize = "0.92rem";
        item.style.fontWeight = "700";
        item.style.color = "var(--primary-deep)";
        item.style.lineHeight = "1.5";
        item.textContent = `${idx + 1}. ${ex}`;
        explorerExamplesList.appendChild(item);
      });
    } catch (err) {
      console.error("Error loading category details:", err);
    }
  }

  if (datasetTabButtons.length > 0) {
    datasetTabButtons.forEach((tab) => {
      tab.addEventListener("click", function () {
        datasetTabButtons.forEach(t => t.classList.remove("active"));
        this.classList.add("active");
        const categoryName = this.getAttribute("data-cat");
        loadCategoryTabDetails(categoryName);
      });
    });

    // Initial load for first tab
    loadCategoryTabDetails("Technology & Science");
  }

  // ==========================================================================
  // 12. Helper UI Handlers
  // ==========================================================================
  function showLoading(show, message = "Analyzing...") {
    if (loadingBox) {
      loadingBox.style.display = show ? "block" : "none";
      const textElem = loadingBox.querySelector(".loading-text");
      if (textElem) textElem.textContent = message;
    }
    if (btnClassify) btnClassify.disabled = show;
    if (btnRandomLoad) btnRandomLoad.disabled = show;
  }

  function showError(msg) {
    if (errorBanner) {
      errorBanner.textContent = msg;
      errorBanner.style.display = "block";
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.style.display = "none";
    }
  }

  // Initialize
  updateDiagnostics();
  renderTimeline();
});
