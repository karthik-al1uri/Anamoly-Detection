const metrics = [
  { label: 'Stream Status', value: 'Connected' },
  { label: 'Active Threshold', value: '0.081' },
  { label: 'Recent Anomalies', value: '12' },
  { label: 'Open Tickets', value: '4' },
];

export default function HomePage() {
  return (
    <main className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Cold Start Streaming Defect Detector</p>
          <h1>Factory-floor anomaly detection for new product launches.</h1>
          <p className="subtitle">
            Monitor streaming inference, review anomaly events, and route maintenance actions.
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

      <section className="panel">
        <h2>Pipeline</h2>
        <div className="pipeline">
          <div className="step">Simulator / Camera Feed</div>
          <div className="step">Databricks Streaming</div>
          <div className="step">PyTorch Autoencoder</div>
          <div className="step">LangGraph Ticketing</div>
          <div className="step">MongoDB Audit Log</div>
        </div>
      </section>
    </main>
  );
}
