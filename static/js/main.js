/**
 * Main JavaScript Application - Industrial Predictive Maintenance Dashboard
 */
document.addEventListener("DOMContentLoaded", () => {

    // ─── DOM References ───────────────────────────────────────────────────────
    const modeTabs           = document.querySelectorAll(".mode-tab");
    const singleSection      = document.getElementById("singlePredictSection");
    const batchSection       = document.getElementById("batchPredictSection");
    const inputCard          = document.getElementById("inputCard");
    const inputCardTitleText = document.getElementById("inputCardTitleText");
    const inputCardTitleIcon = document.querySelector("#inputCardTitle i");
    const liveModeBadge      = document.getElementById("liveModeBadge");
    const modeDescBanner     = document.getElementById("modeDescBanner");
    
    // Sliders & Inputs
    const sensorForm         = document.getElementById("sensorForm");
    const machineType        = document.getElementById("machineType");
    
    const airTemp            = document.getElementById("airTemp");
    const procTemp           = document.getElementById("procTemp");
    const rotSpeed           = document.getElementById("rotSpeed");
    const torque             = document.getElementById("torque");
    const toolWear           = document.getElementById("toolWear");

    const airTempNum         = document.getElementById("airTempNum");
    const procTempNum        = document.getElementById("procTempNum");
    const rotSpeedNum        = document.getElementById("rotSpeedNum");
    const torqueNum          = document.getElementById("torqueNum");
    const toolWearNum        = document.getElementById("toolWearNum");
    
    const airTempVal         = document.getElementById("airTempVal");
    const procTempVal        = document.getElementById("procTempVal");
    const rotSpeedVal        = document.getElementById("rotSpeedVal");
    const torqueVal          = document.getElementById("torqueVal");
    const toolWearVal        = document.getElementById("toolWearVal");
    const submitBtn          = document.getElementById("submitBtn");

    // Output Elements
    const actionBanner       = document.getElementById("actionBanner");
    const bannerIcon         = document.getElementById("bannerIcon");
    const bannerTitle        = document.getElementById("bannerTitle");
    const bannerSubtitle     = document.getElementById("bannerSubtitle");
    const bannerBadge        = document.getElementById("bannerBadge");
    
    const riskVal            = document.getElementById("riskVal");
    const riskLabel          = document.getElementById("riskLabel");
    
    const failurePill        = document.getElementById("failurePill");
    const failureText        = document.getElementById("failureText");
    const failureTypeVal     = document.getElementById("failureTypeVal");
    const confidenceVal      = document.getElementById("confidenceVal");
    
    const downtimeVal        = document.getElementById("downtimeVal");
    const impactVal          = document.getElementById("impactVal");
    const costVal            = document.getElementById("costVal");
    
    const baselineFlagsContainer = document.getElementById("baselineFlagsContainer");
    const diagnosisText      = document.getElementById("diagnosisText");
    const downloadTicketBtn  = document.getElementById("downloadTicketBtn");

    // Batch Ingestion
    const dropzone           = document.getElementById("dropzone");
    const batchFileInput     = document.getElementById("batchFileInput");
    const fleetSummaryGrid   = document.getElementById("fleetSummaryGrid");
    const fleetTableCard     = document.getElementById("fleetTableCard");
    const fleetTableBody     = document.getElementById("fleetTableBody");
    const exportBatchCsvBtn  = document.getElementById("exportBatchCsvBtn");

    // Global State
    let currentMode = "whatif"; // "whatif", "manual", "batch"
    let riskGaugeChart = null;
    let shapChart = null;
    let forecastChart = null;
    let lastPredictionData = null;
    let debounceTimer = null;

    // Map input pairs for bidirectional syncing
    const inputPairs = [
        { slider: airTemp, num: airTempNum, display: airTempVal, unit: "K", decimals: 1 },
        { slider: procTemp, num: procTempNum, display: procTempVal, unit: "K", decimals: 1 },
        { slider: rotSpeed, num: rotSpeedNum, display: rotSpeedVal, unit: "rpm", decimals: 0 },
        { slider: torque, num: torqueNum, display: torqueVal, unit: "Nm", decimals: 1 },
        { slider: toolWear, num: toolWearNum, display: toolWearVal, unit: "min", decimals: 0 }
    ];

    // ─── Mode Switching ───────────────────────────────────────────────────────
    modeTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            modeTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            currentMode = tab.dataset.mode;
            
            if (currentMode === "batch") {
                singleSection.classList.add("hidden");
                batchSection.classList.remove("hidden");
            } else {
                singleSection.classList.remove("hidden");
                batchSection.classList.add("hidden");
                
                if (currentMode === "whatif") {
                    // What-If Simulator Mode Look & Feel
                    inputCard.className = "card input-card mode-whatif";
                    inputCardTitleIcon.className = "fa-solid fa-sliders text-primary";
                    inputCardTitleText.textContent = "What-If Real-Time Simulator";
                    liveModeBadge.innerHTML = `<i class="fa-solid fa-bolt"></i> Live Syncing`;
                    liveModeBadge.className = "badge badge-info";
                    
                    modeDescBanner.className = "mode-desc-banner mode-desc-whatif";
                    modeDescBanner.innerHTML = `<i class="fa-solid fa-sliders"></i> <strong>What-If Simulator Active:</strong> Drag any slider to simulate live sensor telemetry changes in real-time.`;
                    
                    submitBtn.style.display = "none";
                    triggerRealtimePredict();
                } else {
                    // Manual Diagnostic Entry Mode Look & Feel
                    inputCard.className = "card input-card mode-manual";
                    inputCardTitleIcon.className = "fa-solid fa-pen-to-square text-amber";
                    inputCardTitleText.textContent = "Manual Diagnostic Input";
                    liveModeBadge.innerHTML = `<i class="fa-solid fa-pen"></i> Manual Entry`;
                    liveModeBadge.className = "badge";
                    liveModeBadge.style.backgroundColor = "#fffbe6";
                    liveModeBadge.style.color = "#d97706";
                    liveModeBadge.style.border = "1px solid #fde68a";
                    
                    modeDescBanner.className = "mode-desc-banner mode-desc-manual";
                    modeDescBanner.innerHTML = `<i class="fa-solid fa-pen-to-square"></i> <strong>Manual Entry Active:</strong> Type measured values or adjust sliders, then click 'Run Diagnostic Prediction'.`;
                    
                    submitBtn.style.display = "block";
                }
            }
        });
    });

    // ─── Bidirectional Input Synchronization ───────────────────────────────
    inputPairs.forEach(pair => {
        // Slider moved -> update number input & label display
        pair.slider.addEventListener("input", () => {
            const val = parseFloat(pair.slider.value);
            pair.num.value = pair.decimals === 0 ? Math.round(val) : val.toFixed(pair.decimals);
            pair.display.textContent = `${pair.num.value} ${pair.unit}`;
            
            onInputValueChange();
        });

        // Numeric input changed -> update slider & label display
        pair.num.addEventListener("input", () => {
            let val = parseFloat(pair.num.value);
            if (isNaN(val)) return;
            
            const min = parseFloat(pair.slider.min);
            const max = parseFloat(pair.slider.max);
            val = Math.max(min, Math.min(max, val));
            
            pair.slider.value = val;
            pair.display.textContent = `${pair.decimals === 0 ? Math.round(val) : val.toFixed(pair.decimals)} ${pair.unit}`;
            
            onInputValueChange();
        });
    });

    machineType.addEventListener("change", () => {
        onInputValueChange();
    });

    function onInputValueChange() {
        if (currentMode === "whatif") {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                triggerRealtimePredict();
            }, 80); // Fast 80ms live response
        }
    }

    function syncAllInputValues() {
        inputPairs.forEach(pair => {
            const val = parseFloat(pair.slider.value);
            pair.num.value = pair.decimals === 0 ? Math.round(val) : val.toFixed(pair.decimals);
            pair.display.textContent = `${pair.num.value} ${pair.unit}`;
        });
    }

    sensorForm.addEventListener("submit", (e) => {
        e.preventDefault();
        triggerRealtimePredict();
    });

    // ─── API Prediction Call ──────────────────────────────────────────────────
    async function triggerRealtimePredict() {
        const payload = {
            type: machineType.value,
            air_temp: parseFloat(airTemp.value),
            proc_temp: parseFloat(procTemp.value),
            rot_speed: parseFloat(rotSpeed.value),
            torque: parseFloat(torque.value),
            tool_wear: parseFloat(toolWear.value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (data.success) {
                lastPredictionData = data.prediction;
                updateDashboardUI(data.prediction);
            }
        } catch (err) {
            console.error("Prediction API error:", err);
        }
    }

    // ─── Update UI Dashboard ──────────────────────────────────────────────────
    function updateDashboardUI(pred) {
        // 0. Historical Reliability Tracker Badge
        if (pred.reliability_tracker) {
            const relBadge = document.getElementById("relAccVal");
            if (relBadge) relBadge.textContent = `${pred.reliability_tracker.accuracy_pct}%`;
        }

        // 1. Action Banner
        actionBanner.className = `action-banner banner-${pred.tier}`;
        bannerBadge.textContent = pred.tier_label;
        bannerTitle.textContent = pred.action_banner;
        bannerSubtitle.textContent = pred.tier === 'red' ? 
            "Critical anomaly detected. Immediate machine shutdown recommended." :
            (pred.tier === 'yellow' ? "Elevated wear/stress observed. Schedule routine inspection." : 
            "Telemetry parameters within optimal operational safety thresholds.");

        if (pred.tier === 'red') {
            bannerIcon.className = "fa-solid fa-triangle-exclamation";
        } else if (pred.tier === 'yellow') {
            bannerIcon.className = "fa-solid fa-circle-exclamation";
        } else {
            bannerIcon.className = "fa-solid fa-circle-check";
        }

        // 2. Status & Risk Summary
        riskVal.textContent = `${pred.risk_score}%`;
        riskLabel.textContent = pred.tier_label;
        
        failurePill.className = `status-pill pill-${pred.tier}`;
        failureText.textContent = pred.is_failure ? "MACHINE FAILURE RISK" : "OPTIMAL HEALTH";
        failureTypeVal.textContent = pred.failure_type;
        confidenceVal.textContent = `${pred.confidence}%`;

        // 3. Cost Estimate
        downtimeVal.textContent = pred.cost_estimate.downtime;
        impactVal.textContent = pred.cost_estimate.cost_impact;
        costVal.textContent = pred.cost_estimate.est_cost;

        // 4. Self-Baseline Anomaly Flags
        renderBaselineAnomalies(pred.baseline_anomalies);

        // 5. Plain-English Diagnosis Text
        diagnosisText.textContent = pred.diagnosis;

        // 6. Update Gauges & Visual Charts
        renderRiskGauge(pred.risk_score, pred.tier);
        renderShapChart(pred.shap_scores);
        renderForecastChart(pred.risk_forecast);
    }

    // ─── Render Self-Baseline Anomaly Flags ──────────────────────────────────
    function renderBaselineAnomalies(anomalies) {
        if (!baselineFlagsContainer) return;
        baselineFlagsContainer.innerHTML = "";

        if (!anomalies || anomalies.length === 0) {
            baselineFlagsContainer.innerHTML = `
                <span class="baseline-flag" style="background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0;">
                    <i class="fa-solid fa-circle-check"></i> Baseline Sync: Telemetry within historical machine norm
                </span>
            `;
            return;
        }

        anomalies.forEach(a => {
            const flag = document.createElement("span");
            const isCritical = a.severity === "critical";
            flag.className = `baseline-flag ${isCritical ? 'flag-critical' : 'flag-warning'}`;
            
            const icon = isCritical ? 'fa-solid fa-triangle-exclamation' : 'fa-solid fa-circle-exclamation';
            const devSign = a.deviation_pct > 0 ? '+' : '';
            
            flag.innerHTML = `<i class="${icon}"></i> <strong>${a.feature}:</strong> ${a.current} ${a.unit} (${devSign}${a.deviation_pct}% vs ${a.baseline} ${a.unit} baseline)`;
            baselineFlagsContainer.appendChild(flag);
        });
    }

    // ─── Render Doughnut Risk Gauge ───────────────────────────────────────────
    function renderRiskGauge(score, tier) {
        const canvas = document.getElementById("riskGaugeCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        
        let color = "#10b981"; // green
        if (tier === "yellow") color = "#f59e0b";
        if (tier === "red") color = "#ef4444";

        if (riskGaugeChart) {
            riskGaugeChart.destroy();
        }

        riskGaugeChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: [color, "#e2e8f0"],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "78%",
                plugins: { tooltip: { enabled: false } }
            }
        });
    }

    // ─── Render SHAP Explainability Bar Chart ─────────────────────────────────
    function renderShapChart(shapScores) {
        const canvas = document.getElementById("shapChartCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        
        if (!shapScores) shapScores = {};

        // Feature label formatting
        const labelMap = {
            "Air temperature [K]": "Air Temp",
            "Process temperature [K]": "Process Temp",
            "Rotational speed [rpm]": "Rotational Speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool Wear"
        };

        const rawLabels = Object.keys(shapScores);
        const labels = rawLabels.map(l => labelMap[l] || l);
        const values = Object.values(shapScores);
        
        const bgColors = values.map(val => val >= 0 ? "#ef4444" : "#10b981");

        if (shapChart) {
            shapChart.destroy();
        }

        shapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Risk Impact (%)",
                    data: values,
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const val = context.raw;
                                return val >= 0 ? `+${val}% (Increases Failure Risk)` : `${val}% (Decreases Failure Risk)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "#e2e8f0" },
                        ticks: { font: { family: 'Inter' } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', weight: '500' } }
                    }
                }
            }
        });
    }

    // ─── Render 24-Hour Risk Forecast Line Chart ──────────────────────────────
    function renderForecastChart(forecastData) {
        const canvas = document.getElementById("forecastChartCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");

        if (!forecastData || forecastData.length === 0) return;

        const labels = forecastData.map(item => item.time);
        const riskPoints = forecastData.map(item => item.risk);

        // Determine gradient fill & line stroke color based on max forecasted risk
        const maxRisk = Math.max(...riskPoints);
        let strokeColor = "#10b981";
        let fillColorStart = "rgba(16, 185, 129, 0.25)";
        let fillColorEnd = "rgba(16, 185, 129, 0.0)";

        if (maxRisk >= 70) {
            strokeColor = "#ef4444";
            fillColorStart = "rgba(239, 68, 68, 0.25)";
            fillColorEnd = "rgba(239, 68, 68, 0.0)";
        } else if (maxRisk >= 35) {
            strokeColor = "#f59e0b";
            fillColorStart = "rgba(245, 158, 11, 0.25)";
            fillColorEnd = "rgba(245, 158, 11, 0.0)";
        }

        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, fillColorStart);
        gradient.addColorStop(1, fillColorEnd);

        if (forecastChart) {
            forecastChart.destroy();
        }

        forecastChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Projected Risk (%)",
                    data: riskPoints,
                    borderColor: strokeColor,
                    backgroundColor: gradient,
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: strokeColor,
                    pointHoverRadius: 6,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Risk: ${context.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "#f1f5f9" },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: "#e2e8f0" },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: (val) => `${val}%`
                        }
                    }
                }
            }
        });
    }

    // ─── Download Maintenance Ticket ──────────────────────────────────────────
    downloadTicketBtn.addEventListener("click", async () => {
        if (!lastPredictionData) return;
        
        try {
            const response = await fetch("/api/download_ticket", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    format: "csv",
                    machine_id: "MCH-" + Math.floor(1000 + Math.random() * 9000),
                    risk_score: lastPredictionData.risk_score,
                    failure_type: lastPredictionData.failure_type,
                    tier: lastPredictionData.tier,
                    diagnosis: lastPredictionData.diagnosis,
                    shap_scores: lastPredictionData.shap_scores,
                    raw_inputs: lastPredictionData.raw_inputs
                })
            });
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `maintenance_ticket_${lastPredictionData.raw_inputs.type}_type.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            alert("Error downloading ticket: " + err);
        }
    });

    // ─── Batch CSV Mode Handling ──────────────────────────────────────────────
    batchFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadBatchCsv(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--primary-color)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-hover)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-hover)";
        if (e.dataTransfer.files.length > 0) {
            uploadBatchCsv(e.dataTransfer.files[0]);
        }
    });

    async function uploadBatchCsv(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/predict_batch", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                renderFleetResults(data);
            } else {
                alert("Batch CSV Error: " + data.error);
            }
        } catch (err) {
            alert("Upload failed: " + err);
        }
    }

    function renderFleetResults(data) {
        document.getElementById("totalFleetNum").textContent = data.total_machines;
        document.getElementById("criticalFleetNum").textContent = data.critical_count;
        document.getElementById("warningFleetNum").textContent = data.warning_count;
        document.getElementById("healthyFleetNum").textContent = data.healthy_count;

        fleetSummaryGrid.classList.remove("hidden");
        fleetTableCard.classList.remove("hidden");

        fleetTableBody.innerHTML = "";

        data.machines.forEach((m, idx) => {
            const tr = document.createElement("tr");
            
            let statusBadge = `<span class="badge badge-info">Healthy</span>`;
            if (m.tier === "red") statusBadge = `<span class="status-pill pill-red" style="font-size:0.75rem;padding:2px 8px;">Critical</span>`;
            else if (m.tier === "yellow") statusBadge = `<span class="status-pill pill-yellow" style="font-size:0.75rem;padding:2px 8px;">Warning</span>`;
            else statusBadge = `<span class="status-pill pill-green" style="font-size:0.75rem;padding:2px 8px;">Normal</span>`;

            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td><code>${m.machine_id}</code></td>
                <td><span class="val-display">${m.type}</span></td>
                <td><strong style="color:${m.tier === 'red' ? '#ef4444' : (m.tier === 'yellow' ? '#f59e0b' : '#10b981')}">${m.risk_score}%</strong></td>
                <td>${statusBadge}</td>
                <td>${m.failure_type}</td>
                <td>${m.est_downtime}</td>
                <td><strong>${m.cost_impact}</strong></td>
                <td>
                    <button class="btn btn-sm btn-outline inspect-btn" data-json='${JSON.stringify(m)}'>
                        <i class="fa-solid fa-sliders"></i> Inspect
                    </button>
                </td>
            `;
            fleetTableBody.appendChild(tr);
        });

        // Add inspect listeners
        document.querySelectorAll(".inspect-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const m = JSON.parse(btn.dataset.json);
                // Load into What-if simulator
                machineType.value = m.type;
                airTemp.value = m.air_temp;
                procTemp.value = m.proc_temp;
                rotSpeed.value = m.rot_speed;
                torque.value = m.torque;
                toolWear.value = m.tool_wear;
                
                syncAllInputValues();
                
                // Switch to what-if tab
                document.querySelector('.mode-tab[data-mode="whatif"]').click();
            });
        });
    }

    // Export batch results table as CSV
    exportBatchCsvBtn.addEventListener("click", () => {
        let csv = "Rank,Machine ID,Type,Risk Score (%),Status,Failure Type,Est Downtime,Cost Impact\n";
        const rows = fleetTableBody.querySelectorAll("tr");
        rows.forEach(r => {
            const cols = r.querySelectorAll("td");
            if (cols.length > 0) {
                const rowData = Array.from(cols).slice(0, 8).map(c => `"${c.innerText.replace(/"/g, '""')}"`).join(",");
                csv += rowData + "\n";
            }
        });
        const blob = new Blob([csv], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "fleet_risk_ranking_report.csv";
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    });

    // ─── Initial Kickoff ─────────────────────────────────────────────────────
    syncAllInputValues();
    triggerRealtimePredict();
});
