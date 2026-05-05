(function () {
  "use strict";

  const cfg = window.APP_CONFIG || {};
  const apiBase = (cfg.backendApiUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");
  const currentPage = document.body.dataset.page || "";

  function apiUrl(path) {
    if (!path) return "";
    if (String(path).startsWith("http://") || String(path).startsWith("https://")) {
      return path;
    }
    const normalized = String(path).startsWith("/") ? path : `/${path}`;
    return `${apiBase}${normalized}`;
  }

  async function requestJson(path, options) {
    const res = await fetch(apiUrl(path), options);
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const payload = await res.json();
        detail = payload.detail || payload.message || detail;
      } catch (_) {
        // no-op
      }
      throw new Error(detail);
    }
    return res.json();
  }

  function setSidebarToggle() {
    const btn = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");
    if (!btn || !sidebar) return;
    btn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
    document.addEventListener("click", (event) => {
      if (window.innerWidth > 992) return;
      if (!sidebar.classList.contains("open")) return;
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest("#sidebar") || target.closest("#sidebarToggle")) return;
      sidebar.classList.remove("open");
    });
  }

  function createMetaCard(label, value) {
    const safeValue = value === null || value === undefined || value === "" ? "N/A" : String(value);
    return `
      <div class="meta-card">
        <span class="meta-label">${label}</span>
        <div class="meta-value">${safeValue}</div>
      </div>
    `;
  }

  function createAssetCard(label, url) {
    if (!url) return "";
    return `
      <div class="col-md-6 col-xl-3">
        <article class="asset-card">
          <div class="asset-card-header">${label}</div>
          <a href="${apiUrl(url)}" target="_blank" rel="noopener">
            <img src="${apiUrl(url)}" alt="${label}" loading="lazy" />
          </a>
        </article>
      </div>
    `;
  }

  function setMessage(containerId, text, type) {
    const wrap = document.getElementById(containerId);
    if (!wrap) return;
    let alertBox = wrap.querySelector(".alert");
    if (!alertBox) {
      alertBox = document.createElement("div");
      wrap.prepend(alertBox);
    }
    alertBox.className = `alert alert-${type || "info"}`;
    alertBox.textContent = text;
    wrap.hidden = false;
  }

  function clearMessage(containerId) {
    const wrap = document.getElementById(containerId);
    if (wrap) wrap.hidden = true;
  }

  function initUploadPage() {
    const form = document.getElementById("uploadForm");
    if (!form) return;

    const dropZone = document.getElementById("uploadDropZone");
    const fileInput = document.getElementById("scanFile");
    const fileName = document.getElementById("selectedFileName");
    const preview = document.getElementById("uploadPreview");
    const previewPlaceholder = document.getElementById("previewPlaceholder");
    const loader = document.getElementById("uploadLoader");
    const resultSection = document.getElementById("uploadResultSection");
    const metaGrid = document.getElementById("resultMeta");
    const assetGallery = document.getElementById("assetGallery");
    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceText = document.getElementById("confidenceText");
    const classProbabilityGrid = document.getElementById("classProbabilityGrid");
    const submitBtn = document.getElementById("analyzeBtn");

    const allowed = [".png", ".jpg", ".jpeg", ".nii", ".nii.gz", ".dcm"];

    function fileExtension(name) {
      const lower = (name || "").toLowerCase();
      if (lower.endsWith(".nii.gz")) return ".nii.gz";
      const idx = lower.lastIndexOf(".");
      return idx >= 0 ? lower.slice(idx) : "";
    }

    function updatePreview(file) {
      if (!file || !preview || !previewPlaceholder) return;
      const ext = fileExtension(file.name);
      if (ext === ".png" || ext === ".jpg" || ext === ".jpeg") {
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.hidden = false;
        previewPlaceholder.hidden = true;
      } else {
        preview.removeAttribute("src");
        preview.hidden = true;
        previewPlaceholder.hidden = false;
      }
    }

    function onFileSelected(file) {
      if (!file) return;
      const ext = fileExtension(file.name);
      if (!allowed.includes(ext)) {
        setMessage("uploadResultSection", "Unsupported file type. Please upload PNG/JPG/NIfTI/DICOM.", "danger");
        return;
      }
      if (fileName) fileName.textContent = file.name;
      updatePreview(file);
    }

    if (fileInput) {
      fileInput.addEventListener("change", () => {
        const file = fileInput.files && fileInput.files[0];
        onFileSelected(file);
      });
    }

    if (dropZone && fileInput) {
      ["dragenter", "dragover"].forEach((evt) =>
        dropZone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropZone.classList.add("is-dragover");
        })
      );
      ["dragleave", "drop"].forEach((evt) =>
        dropZone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropZone.classList.remove("is-dragover");
        })
      );
      dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const file = dt && dt.files && dt.files[0];
        if (!file) return;
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        onFileSelected(file);
      });
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearMessage("uploadMessage");

      if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        setMessage("uploadMessage", "Please select an MRI file before starting analysis.", "warning");
        return;
      }

      const payload = new FormData(form);

      try {
        if (loader) loader.hidden = false;
        if (submitBtn) submitBtn.setAttribute("disabled", "disabled");

        const result = await requestJson("/api/scans/upload", {
          method: "POST",
          body: payload,
        });

        const confidencePct = Math.round(Number(result.confidence_score || 0) * 100);
        if (confidenceBar) confidenceBar.style.width = `${confidencePct}%`;
        if (confidenceText) confidenceText.textContent = `Confidence: ${confidencePct}%`;

        if (metaGrid) {
          metaGrid.innerHTML = [
            createMetaCard("Patient ID", result.patient_id),
            createMetaCard("Scan ID", result.scan_id),
            createMetaCard("Tumor Detected", result.tumor_detected ? "Yes" : "No"),
            createMetaCard("Tumor Type", result.tumor_type),
            createMetaCard("Stage", result.stage_label),
            createMetaCard("Stage Method", result.stage_method),
            createMetaCard("Risk Category", result.risk_category),
            createMetaCard(
              "Patient Match",
              result.matched_existing_patient ? "Matched Existing Patient" : "New Patient Record"
            ),
            createMetaCard("Match Strategy", result.patient_match_strategy),
            createMetaCard("Runtime Mode", result.runtime_mode),
            createMetaCard("Model Version", result.model_version),
          ].join("");
        }

        if (classProbabilityGrid) {
          const probs = Array.isArray(result.class_probabilities) ? result.class_probabilities : [];
          classProbabilityGrid.innerHTML = probs
            .map((p) => createMetaCard(p.class_name || "Unknown", `${Math.round(Number(p.probability || 0) * 10000) / 100}%`))
            .join("");
        }

        if (assetGallery) {
          assetGallery.innerHTML = [
            createAssetCard("Uploaded MRI", result.image_url),
            createAssetCard("Segmentation Mask", result.mask_url),
            createAssetCard("Grad-CAM Heatmap", result.gradcam_url),
            createAssetCard("Overlay", result.overlay_url),
          ].join("");
        }

        if (resultSection) resultSection.hidden = false;
      } catch (err) {
        setMessage("uploadMessage", err.message || "Failed to analyze MRI.", "danger");
      } finally {
        if (loader) loader.hidden = true;
        if (submitBtn) submitBtn.removeAttribute("disabled");
      }
    });
  }

  async function initHistoryPage() {
    const patientSelect = document.getElementById("patientSelect");
    const patientIdInput = document.getElementById("patientIdInput");
    const loadBtn = document.getElementById("loadHistoryBtn");
    const section = document.getElementById("historySection");
    const summary = document.getElementById("patientSummary");
    const timeline = document.getElementById("historyTimeline");
    if (!patientSelect || !loadBtn || !summary || !timeline || !section) return;

    try {
      const patients = await requestJson("/api/patients");
      patientSelect.innerHTML = `<option value="">Select patient</option>` + patients
        .map((p) => `<option value="${p.patient_id}">${p.patient_id} - ${p.name}</option>`)
        .join("");
    } catch (err) {
      setMessage("historyMessage", err.message || "Unable to load patients.", "danger");
      return;
    }

    loadBtn.addEventListener("click", async () => {
      clearMessage("historyMessage");
      const patientId = (patientIdInput && patientIdInput.value.trim()) || patientSelect.value;
      if (!patientId) {
        setMessage("historyMessage", "Please select or enter a patient ID.", "warning");
        return;
      }

      try {
        const payload = await requestJson(`/api/patients/${encodeURIComponent(patientId)}`);
        const patient = payload.patient || {};
        const scans = payload.scans || [];

        summary.innerHTML = [
          createMetaCard("Patient ID", patient.patient_id),
          createMetaCard("Name", patient.name),
          createMetaCard("Age", patient.age),
          createMetaCard("Gender", patient.gender),
          createMetaCard("Total Scans", scans.length),
        ].join("");

        timeline.innerHTML = scans
          .map((entry) => {
            const scan = entry.scan || {};
            const assets = entry.assets || {};
            return `
              <article class="glass-card info-card timeline-card">
                <h3>${scan.scan_date || "Unknown date"} - ${scan.tumor_type || "N/A"}</h3>
                <p>Detected: ${scan.tumor_detected ? "Yes" : "No"} | Stage: ${scan.stage_label || "N/A"} | Risk: ${scan.risk_category || "N/A"}</p>
                <div class="d-flex flex-wrap gap-2">
                  ${assets.report_url ? `<a class="btn btn-sm btn-outline-light" target="_blank" rel="noopener" href="${apiUrl(assets.report_url)}">Report</a>` : ""}
                  ${assets.overlay_url ? `<a class="btn btn-sm btn-outline-info" target="_blank" rel="noopener" href="${apiUrl(assets.overlay_url)}">Overlay</a>` : ""}
                  ${assets.gradcam_url ? `<a class="btn btn-sm btn-outline-info" target="_blank" rel="noopener" href="${apiUrl(assets.gradcam_url)}">Grad-CAM</a>` : ""}
                </div>
              </article>
            `;
          })
          .join("");

        section.hidden = false;
      } catch (err) {
        setMessage("historyMessage", err.message || "Failed to load patient history.", "danger");
      }
    });
  }

  async function initComparisonPage() {
    const patientSelect = document.getElementById("comparePatientSelect");
    const prevSelect = document.getElementById("previousScanSelect");
    const currSelect = document.getElementById("currentScanSelect");
    const compareBtn = document.getElementById("compareBtn");
    const section = document.getElementById("comparisonSection");
    const metrics = document.getElementById("comparisonMetrics");
    const assets = document.getElementById("comparisonAssets");
    const pdfLink = document.getElementById("comparisonPdfLink");
    if (!patientSelect || !prevSelect || !currSelect || !compareBtn || !section || !metrics || !assets || !pdfLink) return;

    let scanList = [];

    async function loadPatients() {
      const patients = await requestJson("/api/patients");
      patientSelect.innerHTML = `<option value="">Select patient</option>` + patients
        .map((p) => `<option value="${p.patient_id}">${p.patient_id} - ${p.name}</option>`)
        .join("");
    }

    async function loadScans(patientId) {
      const scans = await requestJson(`/api/patients/${encodeURIComponent(patientId)}/scans`);
      scanList = scans || [];
      const opts = `<option value="">Select scan</option>` +
        scanList
          .map((entry) => {
            const s = entry.scan || {};
            return `<option value="${s.id}">${s.id} | ${s.scan_date} | ${s.tumor_type}</option>`;
          })
          .join("");
      prevSelect.innerHTML = opts;
      currSelect.innerHTML = opts;
    }

    try {
      await loadPatients();
    } catch (err) {
      setMessage("comparisonMessage", err.message || "Unable to load patient list.", "danger");
      return;
    }

    patientSelect.addEventListener("change", async () => {
      const patientId = patientSelect.value;
      if (!patientId) return;
      try {
        await loadScans(patientId);
      } catch (err) {
        setMessage("comparisonMessage", err.message || "Unable to load scans.", "danger");
      }
    });

    compareBtn.addEventListener("click", async () => {
      clearMessage("comparisonMessage");
      const patientId = patientSelect.value;
      const previousScanId = Number(prevSelect.value);
      const currentScanId = Number(currSelect.value);
      if (!patientId || !previousScanId || !currentScanId) {
        setMessage("comparisonMessage", "Please select patient and both scans.", "warning");
        return;
      }
      if (previousScanId === currentScanId) {
        setMessage("comparisonMessage", "Choose two different scans for comparison.", "warning");
        return;
      }

      try {
        const payload = await requestJson("/api/scans/compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: patientId,
            previous_scan_id: previousScanId,
            current_scan_id: currentScanId,
          }),
        });

        metrics.innerHTML = [
          createMetaCard("Progression", payload.progression_status),
          createMetaCard("Stage Change", payload.stage_change),
          createMetaCard("Abs Change", payload.absolute_change),
          createMetaCard("% Change", payload.percentage_change),
          createMetaCard("Tumor Type Change", payload.tumor_type_change),
          createMetaCard("Risk Change", payload.risk_level_change),
        ].join("");

        const prevAssets = payload.previous_scan_assets || {};
        const currAssets = payload.current_scan_assets || {};
        assets.innerHTML = [
          createAssetCard("Previous Overlay", prevAssets.overlay_url),
          createAssetCard("Current Overlay", currAssets.overlay_url),
          createAssetCard("Progression Chart", payload.progression_chart_url),
          createAssetCard("Growth Map", payload.growth_map_url),
        ].join("");

        pdfLink.href = apiUrl(
          `/api/reports/comparison/${encodeURIComponent(patientId)}/${previousScanId}/${currentScanId}`
        );

        section.hidden = false;
      } catch (err) {
        setMessage("comparisonMessage", err.message || "Comparison failed.", "danger");
      }
    });
  }

  async function initReportsPage() {
    const patientIdInput = document.getElementById("reportPatientId");
    const filterBtn = document.getElementById("filterReportsBtn");
    const table = document.getElementById("reportsTable");
    if (!filterBtn || !table) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;

    async function loadReports() {
      clearMessage("reportsMessage");
      const patientId = (patientIdInput && patientIdInput.value.trim()) || "";
      const qs = patientId ? `?patient_id=${encodeURIComponent(patientId)}` : "";
      const payload = await requestJson(`/api/reports${qs}`);
      const items = payload.items || [];
      tbody.innerHTML = items
        .map(
          (item) => `
            <tr>
              <td>${item.scan_id}</td>
              <td>${item.patient_id || "N/A"}</td>
              <td>${item.patient_name || "N/A"}</td>
              <td>${item.scan_date || "N/A"}</td>
              <td>${item.tumor_type || "N/A"}</td>
              <td>${item.stage_label || "N/A"}</td>
              <td>${item.risk_category || "N/A"}</td>
              <td><a class="btn btn-sm btn-outline-light" target="_blank" rel="noopener" href="${apiUrl(item.report_url)}">PDF</a></td>
            </tr>
          `
        )
        .join("");
      if (!items.length) {
        setMessage("reportsMessage", "No reports found for the selected filter.", "info");
      }
    }

    filterBtn.addEventListener("click", async () => {
      try {
        await loadReports();
      } catch (err) {
        setMessage("reportsMessage", err.message || "Unable to fetch reports.", "danger");
      }
    });

    try {
      await loadReports();
    } catch (err) {
      setMessage("reportsMessage", err.message || "Unable to fetch reports.", "danger");
    }
  }

  setSidebarToggle();
  if (currentPage === "upload") initUploadPage();
  if (currentPage === "history") initHistoryPage();
  if (currentPage === "comparison") initComparisonPage();
  if (currentPage === "reports") initReportsPage();
})();
