import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

// ─── Icons ───────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 20, color = "currentColor" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);

const icons = {
  upload: "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",
  video: "M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z",
  check: "M20 6L9 17l-5-5",
  x: "M18 6L6 18M6 6l12 12",
  clock: "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 6v6l4 2",
  file: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M16 13H8M16 17H8M10 9H8",
  download: "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3",
  tag: "M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82zM7 7h.01",
  text: "M17 6.1H3M21 12.1H3M15.1 18H3",
  cpu: "M9 2H15M9 22H15M2 9V15M22 9V15M9 9h6v6H9zM2 2l3 3M22 2l-3 3M2 22l3-3M22 22l-3-3",
  refresh: "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15",
};

// ─── Utils ───────────────────────────────────────────────────────────────────
const fmtDuration = (s) => {
  if (!s) return "--:--";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0 ? `${h}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}` : `${m}:${String(sec).padStart(2,"0")}`;
};

const fmtTime = (s) => {
  if (!s) return "";
  return new Date(s * 1000).toLocaleTimeString();
};

const statusColors = {
  pending:    { bg: "#1a1a2e", border: "#3a3a5e", text: "#7070aa" },
  processing: { bg: "#0d1f0d", border: "#2d5a2d", text: "#5aaa5a" },
  done:       { bg: "#0d1d2e", border: "#1a5aaa", text: "#4a9ee8" },
  failed:     { bg: "#2e0d0d", border: "#aa2d2d", text: "#e84a4a" },
};

const stepLabels = { transcription: "Transcription", summarization: "Résumé LLM", ocr: "OCR Frames" };

// ─── Components ──────────────────────────────────────────────────────────────

function ProgressBar({ value, color = "#4a9ee8" }) {
  return (
    <div style={{ height: 3, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" }}>
      <div style={{
        height: "100%", width: `${value}%`,
        background: color,
        transition: "width 0.3s ease",
        boxShadow: `0 0 8px ${color}80`,
      }} />
    </div>
  );
}

function StatusBadge({ status }) {
  const c = statusColors[status] || statusColors.pending;
  const labels = { pending: "En attente", processing: "En cours", done: "Terminé", failed: "Erreur" };
  return (
    <span style={{
      padding: "2px 10px", borderRadius: 20, fontSize: 11, fontFamily: "monospace",
      fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
    }}>
      {labels[status] || status}
    </span>
  );
}

function DropZone({ onFile, disabled }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: `2px dashed ${drag ? "#4a9ee8" : "#2a2a3a"}`,
        borderRadius: 8, padding: "40px 24px", textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.2s",
        background: drag ? "#0a1520" : "transparent",
        boxShadow: drag ? "inset 0 0 24px #4a9ee820" : "none",
      }}
    >
      <input
        ref={inputRef} type="file"
        accept=".mp4,.mkv,.avi,.mov,.webm,.flv,.ts,.m4v"
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      <div style={{ marginBottom: 12, opacity: 0.4 }}>
        <Icon d={icons.upload} size={40} />
      </div>
      <div style={{ color: "#8888aa", fontSize: 14, fontFamily: "monospace" }}>
        Glisser une vidéo ici ou <span style={{ color: "#4a9ee8" }}>cliquer</span>
      </div>
      <div style={{ color: "#444466", fontSize: 11, marginTop: 6, fontFamily: "monospace" }}>
        MP4 · MKV · AVI · MOV · WebM
      </div>
    </div>
  );
}

function JobCard({ job, onSelect, selected }) {
  const c = statusColors[job.status] || statusColors.pending;
  return (
    <div
      onClick={() => onSelect(job)}
      style={{
        padding: "14px 16px", borderRadius: 6, cursor: "pointer",
        border: `1px solid ${selected ? "#4a9ee8" : c.border}`,
        background: selected ? "#0d1f35" : c.bg,
        transition: "all 0.15s",
        marginBottom: 8,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <span style={{ color: "#dde", fontFamily: "monospace", fontSize: 13, fontWeight: 700, wordBreak: "break-all" }}>
          {job.video_name}
        </span>
        <StatusBadge status={job.status} />
      </div>
      {job.status === "processing" && job.step && (
        <div>
          <div style={{ color: "#556677", fontSize: 11, fontFamily: "monospace", marginBottom: 4 }}>
            {stepLabels[job.step] || job.step} — {job.step_progress ?? 0}%
          </div>
          <ProgressBar value={job.step_progress ?? 0} />
        </div>
      )}
      {job.status === "failed" && (
        <div style={{ color: "#e84a4a", fontSize: 11, fontFamily: "monospace", marginTop: 4 }}>
          {job.error?.slice(0, 80)}
        </div>
      )}
    </div>
  );
}

function ResultPanel({ job }) {
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState("summary");

  useEffect(() => {
    if (job?.status === "done") {
      fetch(`${API}/jobs/${job.job_id}/result`)
        .then((r) => r.json())
        .then(setResult)
        .catch(console.error);
    } else {
      setResult(null);
    }
  }, [job]);

  if (!job) return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ color: "#333355", fontFamily: "monospace", fontSize: 14, textAlign: "center" }}>
        <Icon d={icons.video} size={48} color="#333355" />
        <div style={{ marginTop: 16 }}>Sélectionner un job pour voir les résultats</div>
      </div>
    </div>
  );

  const tabs = ["summary", "transcript", "ocr"];
  const tabLabels = { summary: "Résumé", transcript: "Transcription", ocr: "OCR" };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "16px 24px", borderBottom: "1px solid #1a1a2a" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <Icon d={icons.video} color="#4a9ee8" />
          <span style={{ color: "#dde", fontFamily: "monospace", fontWeight: 700, fontSize: 15 }}>
            {job.video_name}
          </span>
          <StatusBadge status={job.status} />
        </div>
        {job.status === "processing" && (
          <div>
            <div style={{ color: "#4a9ee8", fontSize: 12, fontFamily: "monospace", marginBottom: 6 }}>
              {stepLabels[job.step] || "Initialisation"}… {job.step_progress ?? 0}%
            </div>
            <ProgressBar value={job.step_progress ?? 0} />
          </div>
        )}
      </div>

      {/* Tabs */}
      {result && (
        <>
          <div style={{ display: "flex", borderBottom: "1px solid #1a1a2a", padding: "0 24px" }}>
            {tabs.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{
                  padding: "10px 16px", border: "none", background: "none", cursor: "pointer",
                  fontFamily: "monospace", fontSize: 12, letterSpacing: "0.06em",
                  color: tab === t ? "#4a9ee8" : "#556688",
                  borderBottom: `2px solid ${tab === t ? "#4a9ee8" : "transparent"}`,
                  transition: "all 0.15s",
                  textTransform: "uppercase",
                }}
              >
                {tabLabels[t]}
              </button>
            ))}
            <div style={{ flex: 1 }} />
            <a
              href={`${API}/jobs/${job.job_id}/download/result.json`}
              download
              style={{
                display: "flex", alignItems: "center", gap: 6,
                color: "#4a9ee8", textDecoration: "none", fontSize: 11, fontFamily: "monospace",
                alignSelf: "center", padding: "4px 8px", borderRadius: 4,
                border: "1px solid #1a3a5a",
              }}
            >
              <Icon d={icons.download} size={14} /> JSON
            </a>
          </div>

          <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
            {tab === "summary" && result.summary && <SummaryTab summary={result.summary} transcription={result.transcription} />}
            {tab === "transcript" && result.transcription && <TranscriptTab transcription={result.transcription} jobId={job.job_id} videoName={job.video_name} />}
            {tab === "ocr" && result.ocr && <OCRTab ocr={result.ocr} />}
          </div>
        </>
      )}
    </div>
  );
}

function SummaryTab({ summary, transcription }) {
  return (
    <div style={{ maxWidth: 700 }}>
      <h2 style={{ color: "#dde", fontFamily: "'Georgia', serif", fontWeight: 400, fontSize: 22, marginBottom: 4, lineHeight: 1.3 }}>
        {summary.title}
      </h2>
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        <span style={{ background: "#0d1f35", border: "1px solid #1a4060", color: "#4a9ee8", padding: "2px 10px", borderRadius: 20, fontSize: 11, fontFamily: "monospace" }}>
          {summary.category}
        </span>
        {summary.keywords?.map((kw) => (
          <span key={kw} style={{ background: "#1a1a2a", border: "1px solid #2a2a4a", color: "#8888bb", padding: "2px 10px", borderRadius: 20, fontSize: 11, fontFamily: "monospace" }}>
            {kw}
          </span>
        ))}
      </div>

      <Section title="Description" icon={icons.file}>
        <p style={{ color: "#aab", lineHeight: 1.7, fontSize: 14, margin: 0 }}>{summary.description}</p>
      </Section>

      <Section title="Résumé" icon={icons.text}>
        <p style={{ color: "#aab", lineHeight: 1.7, fontSize: 14, margin: 0 }}>{summary.summary}</p>
      </Section>

      {summary.chapters?.length > 0 && (
        <Section title={`Chapitres (${summary.chapters.length})`} icon={icons.clock}>
          {summary.chapters.map((ch, i) => (
            <div key={i} style={{ marginBottom: 16, paddingLeft: 16, borderLeft: "2px solid #1a3a5a" }}>
              <div style={{ display: "flex", gap: 10, alignItems: "baseline", marginBottom: 4 }}>
                <span style={{ color: "#4a9ee8", fontFamily: "monospace", fontSize: 11 }}>
                  {fmtDuration(ch.start)} → {fmtDuration(ch.end)}
                </span>
                <span style={{ color: "#ccd", fontWeight: 700, fontSize: 13 }}>{ch.title}</span>
              </div>
              <p style={{ color: "#889", fontSize: 13, margin: 0, lineHeight: 1.6 }}>{ch.description}</p>
            </div>
          ))}
        </Section>
      )}

      {transcription && (
        <div style={{ marginTop: 24, padding: 12, background: "#0d0d15", borderRadius: 6, border: "1px solid #1a1a2a" }}>
          <div style={{ display: "flex", gap: 20 }}>
            <Stat label="Durée" value={fmtDuration(transcription.duration)} />
            <Stat label="Langue" value={transcription.language?.toUpperCase()} />
            <Stat label="Segments" value={transcription.segments?.length} />
          </div>
        </div>
      )}
    </div>
  );
}

function TranscriptTab({ transcription, jobId, videoName }) {
  const [search, setSearch] = useState("");
  const segments = transcription?.segments || [];
  const filtered = search
    ? segments.filter((s) => s.text.toLowerCase().includes(search.toLowerCase()))
    : segments;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center" }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Rechercher dans la transcription..."
          style={{
            flex: 1, padding: "8px 12px", background: "#0d0d18", border: "1px solid #2a2a3a",
            borderRadius: 6, color: "#ccd", fontFamily: "monospace", fontSize: 13,
            outline: "none",
          }}
        />
        <a
          href={`${API}/jobs/${jobId}/download/${videoName}.srt`}
          download
          style={{
            display: "flex", alignItems: "center", gap: 6, color: "#4a9ee8",
            textDecoration: "none", fontSize: 11, fontFamily: "monospace",
            padding: "8px 12px", borderRadius: 6, border: "1px solid #1a3a5a",
            whiteSpace: "nowrap",
          }}
        >
          <Icon d={icons.download} size={14} /> SRT
        </a>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {filtered.map((seg, i) => (
          <div key={i} style={{
            display: "flex", gap: 14, padding: "8px 12px", borderRadius: 4,
            background: "#0a0a12", border: "1px solid #14141e",
          }}>
            <span style={{ color: "#4a9ee8", fontFamily: "monospace", fontSize: 11, whiteSpace: "nowrap", marginTop: 2 }}>
              {fmtDuration(seg.start)}
            </span>
            <span style={{ color: "#ccd", fontSize: 13, lineHeight: 1.6 }}>{seg.text}</span>
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ color: "#444466", fontFamily: "monospace", textAlign: "center", padding: 40 }}>
            Aucun résultat
          </div>
        )}
      </div>
    </div>
  );
}

function OCRTab({ ocr }) {
  const entries = ocr?.entries || [];
  const grouped = {};
  entries.forEach((e) => {
    const key = e.text;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(e.timecode);
  });

  return (
    <div>
      <div style={{ marginBottom: 20, padding: 12, background: "#0d0d15", borderRadius: 6, border: "1px solid #1a1a2a" }}>
        <div style={{ display: "flex", gap: 20 }}>
          <Stat label="Frames analysées" value={ocr.total_frames_sampled} />
          <Stat label="Frames avec texte" value={ocr.frames_with_text} />
          <Stat label="Textes uniques" value={Object.keys(grouped).length} />
        </div>
      </div>

      {entries.length === 0 ? (
        <div style={{ color: "#444466", fontFamily: "monospace", textAlign: "center", padding: 60 }}>
          Aucun texte détecté dans les frames
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {entries.map((entry, i) => (
            <div key={i} style={{
              display: "flex", gap: 14, padding: "8px 12px", borderRadius: 4,
              background: "#0a0a12", border: "1px solid #14141e", alignItems: "flex-start",
            }}>
              <span style={{ color: "#e8a44a", fontFamily: "monospace", fontSize: 11, whiteSpace: "nowrap", marginTop: 2 }}>
                {fmtDuration(entry.timecode)}
              </span>
              <span style={{ color: "#ccd", fontSize: 13, lineHeight: 1.6 }}>{entry.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Section({ title, icon, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Icon d={icon} size={14} color="#4a9ee8" />
        <span style={{ color: "#778899", fontFamily: "monospace", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase" }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <div style={{ color: "#445566", fontFamily: "monospace", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</div>
      <div style={{ color: "#aabbcc", fontFamily: "monospace", fontSize: 14, fontWeight: 700 }}>{value ?? "—"}</div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const wsRefs = useRef({});

  // Poll jobs list
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API}/jobs`);
        const data = await r.json();
        setJobs(data);
        // Auto-select if only one
        if (data.length === 1 && !selected) setSelected(data[0]);
      } catch {}
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket for active jobs
  useEffect(() => {
    jobs.forEach((job) => {
      if (job.status === "processing" && !wsRefs.current[job.job_id]) {
        const ws = new WebSocket(`ws://localhost:8000/ws/${job.job_id}`);
        wsRefs.current[job.job_id] = ws;
        ws.onmessage = (e) => {
          const update = JSON.parse(e.data);
          setJobs((prev) => prev.map((j) => j.job_id === update.job_id ? { ...j, ...update } : j));
          setSelected((prev) => prev?.job_id === update.job_id ? { ...prev, ...update } : prev);
        };
        ws.onclose = () => { delete wsRefs.current[job.job_id]; };
      }
    });
  }, [jobs]);

  // Check system status
  useEffect(() => {
    fetch(`${API}/models/status`).then((r) => r.json()).then(setSystemStatus).catch(() => {});
  }, []);

  const handleFile = async (file) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const r = await fetch(`${API}/jobs/upload`, { method: "POST", body: formData });
      const job = await r.json();
      setJobs((prev) => [job, ...prev]);
      setSelected(job);
    } catch (e) {
      alert("Erreur upload: " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (jobId, e) => {
    e.stopPropagation();
    await fetch(`${API}/jobs/${jobId}`, { method: "DELETE" });
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    if (selected?.job_id === jobId) setSelected(null);
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#060608",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex", flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "0 24px", height: 52, borderBottom: "1px solid #12121e",
        display: "flex", alignItems: "center", gap: 16,
        background: "#080810",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, background: "linear-gradient(135deg, #1a4a8a, #0d2a5a)",
            borderRadius: 6, border: "1px solid #2a5aaa",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Icon d={icons.cpu} size={14} color="#4a9ee8" />
          </div>
          <span style={{ color: "#ccd", fontSize: 13, fontWeight: 700, letterSpacing: "0.06em" }}>
            VIDEO<span style={{ color: "#4a9ee8" }}>PIPELINE</span>
          </span>
        </div>

        <div style={{ flex: 1 }} />

        {systemStatus && (
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <StatusDot ok={systemStatus.ollama} label={`LLM: ${systemStatus.ollama_model}`} />
            <StatusDot ok={true} label={`ASR: ${systemStatus.whisper_model}`} />
          </div>
        )}
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Sidebar */}
        <div style={{
          width: 300, borderRight: "1px solid #12121e",
          display: "flex", flexDirection: "column",
          background: "#07070e",
        }}>
          <div style={{ padding: 16, borderBottom: "1px solid #12121e" }}>
            <DropZone onFile={handleFile} disabled={uploading} />
            {uploading && (
              <div style={{ color: "#4a9ee8", fontSize: 11, textAlign: "center", marginTop: 8, animation: "pulse 1s infinite" }}>
                Envoi en cours…
              </div>
            )}
          </div>

          <div style={{ flex: 1, overflow: "auto", padding: 12 }}>
            <div style={{ color: "#334455", fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 10, paddingLeft: 4 }}>
              Jobs ({jobs.length})
            </div>
            {jobs.length === 0 ? (
              <div style={{ color: "#222233", fontSize: 12, textAlign: "center", padding: "40px 16px" }}>
                Aucun job
              </div>
            ) : (
              jobs.map((job) => (
                <div key={job.job_id} style={{ position: "relative" }}>
                  <JobCard job={job} onSelect={setSelected} selected={selected?.job_id === job.job_id} />
                  <button
                    onClick={(e) => handleDelete(job.job_id, e)}
                    style={{
                      position: "absolute", top: 10, right: 10,
                      background: "none", border: "none", cursor: "pointer",
                      color: "#334455", padding: 2, display: "flex",
                    }}
                    title="Supprimer"
                  >
                    <Icon d={icons.x} size={12} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Main panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <ResultPanel job={selected} />
        </div>
      </div>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #080810; }
        ::-webkit-scrollbar-thumb { background: #1a2a3a; border-radius: 3px; }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
      `}</style>
    </div>
  );
}

function StatusDot({ ok, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 6, height: 6, borderRadius: "50%",
        background: ok ? "#4aaa4a" : "#aa4a4a",
        boxShadow: `0 0 6px ${ok ? "#4aaa4a" : "#aa4a4a"}`,
      }} />
      <span style={{ color: "#445566", fontSize: 10, letterSpacing: "0.06em" }}>{label}</span>
    </div>
  );
}
