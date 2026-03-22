export const dynamic = 'force-dynamic';

type AnomalyEvent = {
  image_path: string;
  source_label?: string;
  mse_score?: number;
  threshold?: number;
  status?: string;
  is_anomaly?: boolean;
  event_ts?: string;
  category?: string;
  threshold_strategy?: string;
};

type TicketPreview = {
  title: string;
  priority: string;
  defect_description: string;
  recommended_action: string;
};

type RetrievedContext = {
  document_id?: string;
  source?: string;
  title?: string;
  content: string;
  score: number;
};

type DiagnosticReport = {
  image_path: string;
  source_label?: string;
  category?: string;
  status: string;
  priority: string;
  anomaly_score?: number;
  threshold?: number;
  threshold_gap?: number;
  defect_description: string;
  analysis_summary: string;
  analysis_source: string;
  recommended_action: string;
  retrieved_context: string;
  retrieved_matches: RetrievedContext[];
  ticket_preview: TicketPreview;
};

function getApiBaseUrl() {
  return process.env.DASHBOARD_API_BASE_URL ?? 'http://localhost:8000';
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T | null> {
  try {
    const response = await fetch(`${getApiBaseUrl()}${path}`, {
      ...init,
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

async function loadDashboardData() {
  const anomaliesPayload = await fetchJson<{ items: AnomalyEvent[] }>('/api/anomalies/recent');
  const anomalies = anomaliesPayload?.items ?? [];
  const latestAnomaly = anomalies[0] ?? null;

  let diagnostic: DiagnosticReport | null = null;
  if (latestAnomaly) {
    diagnostic = await fetchJson<DiagnosticReport>('/api/diagnostics/from-anomaly', {
      method: 'POST',
      body: JSON.stringify(latestAnomaly),
    });
  }

  return { anomalies, latestAnomaly, diagnostic };
}

export default async function HomePage() {
  const { anomalies, latestAnomaly, diagnostic } = await loadDashboardData();
  const anomalyCount = anomalies.filter((item) => item.status === 'anomaly' || item.is_anomaly).length;
  const activeThreshold = latestAnomaly?.threshold ?? diagnostic?.threshold ?? null;
  const metrics = [
    { label: 'Stream Status', value: anomalies.length > 0 ? 'Live' : 'Waiting' },
    { label: 'Active Threshold', value: activeThreshold !== null ? activeThreshold.toFixed(6) : 'N/A' },
    { label: 'Recent Events', value: String(anomalies.length) },
    { label: 'Flagged Anomalies', value: String(anomalyCount) },
  ];

  return (
    <main className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Cold Start Streaming Defect Detector</p>
          <h1>Factory-floor anomaly detection, diagnostics, and ticket drafting.</h1>
          <p className="subtitle">
            Review streaming anomaly events, inspect LLM-backed diagnostics, and preview maintenance actions from one dashboard.
          </p>
        </div>
      </section>

      <section className="grid">
        {metrics.map((metric) => (
          <article className="card" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </article>
        ))}
      </section>

      <section className="panel stack">
        <div className="sectionHeader">
          <h2>Recent anomaly events</h2>
          <p>{anomalies.length > 0 ? 'Latest events from the backend anomaly feed.' : 'No anomaly events available yet.'}</p>
        </div>
        <div className="list">
          {anomalies.length > 0 ? (
            anomalies.slice(0, 8).map((event) => (
              <article className="listItem" key={`${event.image_path}-${event.event_ts ?? 'event'}`}>
                <div>
                  <strong>{event.source_label ?? 'unknown'}</strong>
                  <p>{event.image_path}</p>
                </div>
                <div className="meta">
                  <span className={`badge ${event.status === 'anomaly' || event.is_anomaly ? 'badgeAlert' : 'badgeOk'}`}>
                    {event.status ?? (event.is_anomaly ? 'anomaly' : 'normal')}
                  </span>
                  <span>MSE {event.mse_score?.toFixed(6) ?? 'N/A'}</span>
                </div>
              </article>
            ))
          ) : (
            <div className="emptyState">Start the API and anomaly feed to populate this panel.</div>
          )}
        </div>
      </section>

      <section className="twoColumn">
        <section className="panel stack">
          <div className="sectionHeader">
            <h2>Latest diagnostic</h2>
            <p>{diagnostic ? `Generated via ${diagnostic.analysis_source}.` : 'No diagnostic available yet.'}</p>
          </div>
          {diagnostic ? (
            <>
              <div className="detailBlock">
                <span className="detailLabel">Summary</span>
                <p>{diagnostic.analysis_summary}</p>
              </div>
              <div className="detailBlock">
                <span className="detailLabel">Recommended action</span>
                <p>{diagnostic.recommended_action}</p>
              </div>
              <div className="detailBlock">
                <span className="detailLabel">Retrieved context</span>
                <p>{diagnostic.retrieved_context}</p>
              </div>
            </>
          ) : (
            <div className="emptyState">Diagnostics will appear once the backend can analyze the latest anomaly.</div>
          )}
        </section>

        <section className="panel stack">
          <div className="sectionHeader">
            <h2>Ticket preview</h2>
            <p>{diagnostic ? 'Draft generated from the latest anomaly event.' : 'No ticket preview yet.'}</p>
          </div>
          {diagnostic ? (
            <>
              <div className="ticketHeader">
                <strong>{diagnostic.ticket_preview.title}</strong>
                <span className={`badge ${diagnostic.ticket_preview.priority === 'high' ? 'badgeAlert' : 'badgeOk'}`}>
                  {diagnostic.ticket_preview.priority}
                </span>
              </div>
              <div className="detailBlock">
                <span className="detailLabel">Defect description</span>
                <p>{diagnostic.ticket_preview.defect_description}</p>
              </div>
              <div className="detailBlock">
                <span className="detailLabel">Action</span>
                <p>{diagnostic.ticket_preview.recommended_action}</p>
              </div>
            </>
          ) : (
            <div className="emptyState">Ticket previews will appear after diagnostics are generated.</div>
          )}
        </section>
      </section>

      <section className="panel">
        <h2>Pipeline</h2>
        <div className="pipeline">
          <div className="step">Simulator / Camera Feed</div>
          <div className="step">Databricks Streaming</div>
          <div className="step">PyTorch Autoencoder</div>
          <div className="step">OpenAI + Retrieval Diagnostics</div>
          <div className="step">Dashboard Ticket Preview</div>
        </div>
      </section>
    </main>
  );
}
