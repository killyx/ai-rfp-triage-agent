/**
 * Client-side controller for the Autonomous B2B RFP & Lead Triage Simulator.
 * Uses semantic DOM manipulation, CSS Custom Properties, and native <dialog> APIs.
 */

(() => {
  let activeLead = null;

  const statusPill = document.getElementById("pipeline-status-pill");
  const rawTextarea = document.getElementById("rfp-raw-textarea");
  const textForm = document.getElementById("text-triage-form");
  const pdfInput = document.getElementById("pdf-file-input");

  const extractedJsonCode = document.getElementById("extracted-json-code");
  const extractionModelBadge = document.getElementById("extraction-model-badge");
  const ragTopSimBadge = document.getElementById("rag-top-sim-badge");
  const techVerificationBox = document.getElementById("tech-verification-box");
  const ragCaseStudiesList = document.getElementById("rag-case-studies-list");

  const scoringReasoningText = document.getElementById("scoring-reasoning-text");
  const classificationBadge = document.getElementById("classification-badge");
  const totalScoreBadge = document.getElementById("total-score-badge");
  const scoreProgressBar = document.getElementById("score-progress-bar");
  const scoreBudget = document.getElementById("score-budget");
  const scoreTech = document.getElementById("score-tech");
  const scoreClarity = document.getElementById("score-clarity");
  const scoreTimeline = document.getElementById("score-timeline");
  const cascadeSavings = document.getElementById("cascade-savings");

  const hitlStatusBadge = document.getElementById("hitl-status-badge");
  const hitlLockIndicator = document.getElementById("hitl-lock-indicator");
  const slackTimestamp = document.getElementById("slack-timestamp");
  const slackCardBody = document.getElementById("slack-card-body");
  const proposalPreviewContainer = document.getElementById("proposal-preview-container");
  const proposalSubjectLine = document.getElementById("proposal-subject-line");
  const proposalModelTier = document.getElementById("proposal-model-tier");
  const proposalMarkdownBody = document.getElementById("proposal-markdown-body");
  const hitlActionButtons = document.getElementById("hitl-action-buttons");

  const btnApproveLead = document.getElementById("btn-approve-lead");
  const btnOpenEditDialog = document.getElementById("btn-open-edit-dialog");
  const btnRejectLead = document.getElementById("btn-reject-lead");

  const editDialog = document.getElementById("edit-draft-dialog");
  const editDraftTextarea = document.getElementById("edit-draft-textarea");
  const reviewerNameInput = document.getElementById("reviewer-name-input");
  const btnSaveAndApproveDialog = document.getElementById("btn-save-and-approve-dialog");
  const btnCloseDialogTop = document.getElementById("btn-close-dialog-top");
  const btnCancelDialog = document.getElementById("btn-cancel-dialog");

  const crmAuditTbody = document.getElementById("crm-audit-tbody");

  function setStatus(label, tone = "neutral") {
    statusPill.textContent = label;
    statusPill.className = "px-3 py-1 rounded-full text-xs font-medium border ";
    if (tone === "busy") {
      statusPill.className += "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 animate-pulse";
    } else if (tone === "success") {
      statusPill.className += "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    } else if (tone === "error") {
      statusPill.className += "bg-rose-500/20 text-rose-300 border-rose-500/40";
    } else {
      statusPill.className += "bg-slate-800 text-slate-300 border-slate-700";
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
  }

  function renderLeadRecord(lead) {
    activeLead = lead;
    rawTextarea.value = lead.raw_text || "";

    // 1. Structured Extraction JSON
    extractedJsonCode.textContent = JSON.stringify(lead.extraction, null, 2);
    extractionModelBadge.textContent = `${lead.telemetry.extraction_model} (${lead.telemetry.extraction_tokens} tok)`;

    // 2. RAG Portfolio & Tech Stack Match
    const simPct = Math.round((lead.rag_result.top_similarity_score || 0) * 100);
    ragTopSimBadge.textContent = `Top Similarity: ${simPct}%`;

    const tv = lead.rag_result.tech_verification;
    const supportedBadges = (tv.supported_matches || [])
      .map((t) => `<span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">${escapeHtml(t)}</span>`)
      .join(" ");
    const unsupportedBadges = (tv.unsupported_flagged || [])
      .map((t) => `<span class="px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30">${escapeHtml(t)} (Out of Scope)</span>`)
      .join(" ");

    techVerificationBox.innerHTML = `
      <div class="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
        <div class="flex items-center justify-between text-slate-300">
          <span>Supported Tech Coverage: <strong>${Math.round(tv.coverage_ratio * 100)}%</strong></span>
          <span class="font-mono text-indigo-300">Pod Burn: $${tv.estimated_weekly_burn_usd.toLocaleString()}/wk</span>
        </div>
        <div class="flex flex-wrap gap-1.5">
          ${supportedBadges || '<span class="text-slate-500">No explicit supported stack locked yet</span>'}
          ${unsupportedBadges}
        </div>
      </div>
    `;

    ragCaseStudiesList.innerHTML = (lead.rag_result.matched_case_studies || [])
      .map((cs) => {
        const pct = Math.round(cs.similarity_score * 100);
        return `
          <article class="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
            <div class="flex items-center justify-between gap-2">
              <h3 class="text-xs font-bold text-white">${escapeHtml(cs.title)}</h3>
              <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300">${pct}% match</span>
            </div>
            <p class="text-[11px] text-slate-400">${escapeHtml(cs.industry)} • Budget: $${cs.budget_usd.toLocaleString()} • ${cs.timeline_weeks} wks</p>
            <p class="text-xs text-slate-300">${escapeHtml(cs.summary)}</p>
          </article>
        `;
      })
      .join("");

    // 3. Multi-factor Scoring & CSS Custom Properties
    const sc = lead.scoring;
    scoringReasoningText.textContent = sc.reasoning_summary;
    totalScoreBadge.textContent = `${sc.total_score}/100`;
    scoreBudget.textContent = `${sc.budget_score}/35`;
    scoreTech.textContent = `${sc.tech_stack_score}/30`;
    scoreClarity.textContent = `${sc.scope_clarity_score}/20`;
    scoreTimeline.textContent = `${sc.timeline_realism_score}/15`;
    cascadeSavings.textContent = `${lead.telemetry.cost_savings_percentage}%`;

    const hue = sc.classification === "HIGH_FIT" ? 150 : sc.classification === "AMBIGUOUS" ? 42 : 350;
    scoreProgressBar.style.setProperty("--score-percent", `${sc.total_score}%`);
    scoreProgressBar.style.setProperty("--score-hue", String(hue));

    classificationBadge.textContent = sc.classification;
    if (sc.classification === "HIGH_FIT") {
      classificationBadge.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
    } else if (sc.classification === "AMBIGUOUS") {
      classificationBadge.className = "px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40";
    } else {
      classificationBadge.className = "px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40";
    }

    // 4. Slack Block Kit Simulator & HITL Gate State
    slackTimestamp.textContent = lead.created_at;
    const budgetFormatted = lead.extraction.estimated_budget_usd != null
      ? `$${lead.extraction.estimated_budget_usd.toLocaleString()} USD`
      : "Not Disclosed";
    const timelineFormatted = lead.extraction.timeline_weeks != null
      ? `${lead.extraction.timeline_weeks} Weeks`
      : "Unspecified";

    slackCardBody.innerHTML = `
      <div class="p-3 rounded-lg bg-slate-900/90 border-l-4 ${
        sc.classification === "HIGH_FIT"
          ? "border-emerald-500"
          : sc.classification === "AMBIGUOUS"
          ? "border-amber-500"
          : "border-rose-500"
      } space-y-2">
        <div class="font-bold text-white text-sm">
          [${escapeHtml(sc.classification)}] ${escapeHtml(lead.extraction.company)} — Contact: ${escapeHtml(lead.extraction.client_name)}
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
          <div><span class="text-slate-400">Extracted Budget:</span> <strong class="text-white">${escapeHtml(budgetFormatted)}</strong></div>
          <div><span class="text-slate-400">Timeline:</span> <strong class="text-white">${escapeHtml(timelineFormatted)}</strong></div>
          <div><span class="text-slate-400">CRM Deal ID:</span> <code class="text-indigo-300">${escapeHtml(lead.crm_deal_id)}</code></div>
        </div>
        <div class="text-xs text-slate-300">
          <span class="text-slate-400">Commercial Band:</span> <code class="text-cyan-300">${escapeHtml(lead.proposal_draft.estimated_price_range_usd)}</code>
        </div>
      </div>
    `;

    proposalPreviewContainer.classList.remove("hidden");
    proposalSubjectLine.textContent = `Subject: ${lead.proposal_draft.subject}`;
    proposalModelTier.textContent = `Synthesized via ${lead.telemetry.synthesis_model} (${lead.telemetry.processing_latency_ms} ms)`;
    proposalMarkdownBody.textContent = lead.proposal_draft.body_markdown;

    updateHitlBanner(lead);
  }

  function updateHitlBanner(lead) {
    if (lead.hitl_status === "PENDING_APPROVAL") {
      hitlStatusBadge.textContent = "🔒 DISPATCH LOCKED (AWAITING HITL)";
      hitlStatusBadge.className = "text-xs font-mono px-2.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30";
      hitlLockIndicator.className = "w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping";
      hitlActionButtons.classList.remove("hidden");
    } else if (lead.hitl_status === "APPROVED_AND_SENT") {
      hitlStatusBadge.textContent = `✅ APPROVED & SENT (${lead.approved_by || "Reviewer"})`;
      hitlStatusBadge.className = "text-xs font-mono px-2.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
      hitlLockIndicator.className = "w-2.5 h-2.5 rounded-full bg-emerald-400";
      hitlActionButtons.classList.add("hidden");
    } else {
      hitlStatusBadge.textContent = `🛑 REJECTED & SUPPRESSED (${lead.approved_by || "Reviewer"})`;
      hitlStatusBadge.className = "text-xs font-mono px-2.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30";
      hitlLockIndicator.className = "w-2.5 h-2.5 rounded-full bg-rose-500";
      hitlActionButtons.classList.add("hidden");
    }
  }

  async function refreshAuditLog() {
    try {
      const res = await fetch("/api/crm/audit-log");
      if (!res.ok) return;
      const data = await res.json();
      const entries = data.audit_log || [];
      if (!entries.length) return;

      crmAuditTbody.innerHTML = entries
        .map((e) => {
          const gateBadge = e.outbound_email_dispatched
            ? `<span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">DISPATCHED</span>`
            : `<span class="px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">${escapeHtml(e.hitl_gate_status)}</span>`;
          return `
            <tr class="hover:bg-slate-900/50">
              <td class="py-2.5 pr-3 font-mono text-[11px] text-indigo-300">${escapeHtml(e.entry_id)}</td>
              <td class="py-2.5 pr-3 font-semibold text-white">${escapeHtml(e.company)}</td>
              <td class="py-2.5 pr-3 text-slate-300">${escapeHtml(e.deal_stage)}</td>
              <td class="py-2.5 pr-3">${gateBadge}</td>
              <td class="py-2.5 text-slate-400">${escapeHtml(e.details)}</td>
            </tr>
          `;
        })
        .join("");
    } catch (err) {
      console.error("Failed to refresh CRM audit log:", err);
    }
  }

  async function triggerPresetScenario(presetId) {
    setStatus(`Running pipeline for ${presetId}...`, "busy");
    try {
      const res = await fetch(`/api/triage/sample/${encodeURIComponent(presetId)}`, {
        method: "POST",
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Preset execution failed");
      }
      const record = await res.json();
      renderLeadRecord(record);
      await refreshAuditLog();
      setStatus(`Triaged ${record.extraction.company} (${record.scoring.total_score}/100)`, "success");
    } catch (err) {
      setStatus(`Error: ${err.message}`, "error");
    }
  }

  async function submitHitlAction(action, editedMarkdown = null, notes = null) {
    if (!activeLead) return;
    setStatus(`Executing HITL '${action}' webhook...`, "busy");
    try {
      const res = await fetch("/api/lead/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lead_id: activeLead.lead_id,
          action,
          reviewer_name: reviewerNameInput.value || "Staff Solutions Architect",
          edited_draft_markdown: editedMarkdown,
          reviewer_notes: notes,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "HITL webhook failed");
      }
      const result = await res.json();
      renderLeadRecord(result.lead);
      await refreshAuditLog();
      setStatus(`HITL Gate: ${result.hitl_status}`, "success");
    } catch (err) {
      setStatus(`HITL Error: ${err.message}`, "error");
    }
  }

  // Bind 1-Click Scenario Preset Buttons
  document.querySelectorAll(".preset-trigger").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetId = btn.getAttribute("data-preset-id");
      if (presetId) triggerPresetScenario(presetId);
    });
  });

  // Bind Raw Text Triage Form
  textForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = rawTextarea.value.trim();
    if (text.length < 15) return;
    setStatus("Extracting structured parameters & running RAG...", "busy");
    try {
      const res = await fetch("/api/triage/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: text, source_name: "dashboard_paste.txt" }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to triage text");
      }
      const record = await res.json();
      renderLeadRecord(record);
      await refreshAuditLog();
      setStatus(`Triaged ${record.extraction.company} (${record.scoring.total_score}/100)`, "success");
    } catch (err) {
      setStatus(`Error: ${err.message}`, "error");
    }
  });

  // Bind PDF File Upload
  pdfInput.addEventListener("change", async () => {
    const file = pdfInput.files?.[0];
    if (!file) return;
    setStatus(`Parsing PDF '${file.name}' via PyMuPDF...`, "busy");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("/api/triage/pdf", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "PDF parsing failed");
      }
      const record = await res.json();
      renderLeadRecord(record);
      await refreshAuditLog();
      setStatus(`Parsed & Triaged ${file.name} (${record.scoring.total_score}/100)`, "success");
    } catch (err) {
      setStatus(`PDF Error: ${err.message}`, "error");
    } finally {
      pdfInput.value = "";
    }
  });

  // Bind Slack HITL Action Buttons
  btnApproveLead.addEventListener("click", () => submitHitlAction("approve"));
  btnRejectLead.addEventListener("click", () =>
    submitHitlAction("reject", null, "Rejected by Solutions Architect during Slack HITL triage.")
  );

  btnOpenEditDialog.addEventListener("click", () => {
    if (!activeLead) return;
    editDraftTextarea.value = activeLead.proposal_draft.body_markdown;
    if (typeof editDialog.showModal === "function") {
      editDialog.showModal();
    }
  });

  // Fallback close listeners in addition to native command="close"
  btnCloseDialogTop.addEventListener("click", () => editDialog.close());
  btnCancelDialog.addEventListener("click", () => editDialog.close());

  btnSaveAndApproveDialog.addEventListener("click", async () => {
    const updatedMarkdown = editDraftTextarea.value.trim();
    editDialog.close();
    await submitHitlAction("edit_and_approve", updatedMarkdown, "Edited draft in workbench before dispatch.");
  });

  // Auto-load Scenario 1 on initial page load so the dashboard is immediately populated for demos
  window.addEventListener("DOMContentLoaded", () => {
    triggerPresetScenario("ideal_fintech_pdf");
  });
})();
