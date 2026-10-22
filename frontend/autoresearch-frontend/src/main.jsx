import React, { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BookOpen, Search, LibraryBig, Files, Settings2, ArrowUpRight,
  FileText, Menu, X, Sparkles, Clock3, CircleHelp, RefreshCw,
  Download, CheckCircle2, AlertCircle, LoaderCircle
} from 'lucide-react';
import './styles.css';

const API = 'http://127.0.0.1:8000';

const navItems = [
  { id: 'research', label: 'Research', icon: Search },
  { id: 'papers', label: 'Papers', icon: LibraryBig },
  { id: 'reports', label: 'Reports', icon: Files },
];

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {}
    throw new Error(message);
  }

  return response.json();
}

function App() {
  const [active, setActive] = useState('research');
  const [question, setQuestion] = useState('');
  const [notice, setNotice] = useState('');
  const [filter, setFilter] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [papers, setPapers] = useState([]);
  const [reports, setReports] = useState([]);
  const [runs, setRuns] = useState([]);
  const [currentRun, setCurrentRun] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      await apiRequest('/api/health');
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  const refreshData = useCallback(async () => {
    setRefreshing(true);
    try {
      const [runData, paperData, reportData] = await Promise.all([
        apiRequest('/api/research'),
        apiRequest('/api/papers'),
        apiRequest('/api/reports'),
      ]);
      setRuns(runData);
      setPapers(paperData);
      setReports(reportData);
      setApiOnline(true);
    } catch (error) {
      setApiOnline(false);
      setNotice(`Could not load data: ${error.message}. Check that the backend is running.`);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    refreshData();
  }, [checkHealth, refreshData]);

  useEffect(() => {
    const timer = setInterval(() => {
      checkHealth();
      refreshData();
    }, 5000);
    return () => clearInterval(timer);
  }, [checkHealth, refreshData]);

  useEffect(() => {
    if (!currentRun?.id || !['queued', 'running'].includes(currentRun.status)) {
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const data = await apiRequest(`/api/research/${currentRun.id}`);
        if (cancelled) return;
        setCurrentRun(data);

        if (['completed', 'failed'].includes(data.status)) {
          await refreshData();
          if (!cancelled) {
            setNotice(
              data.status === 'completed'
                ? 'Research completed! Your results and reports are ready.'
                : `Research failed: ${data.error || 'Check the logs below.'}`
            );
          }
        }
      } catch (error) {
        if (!cancelled) setNotice(`Unable to check research progress: ${error.message}`);
      }
    };

    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [currentRun?.id, currentRun?.status, refreshData]);

  async function startResearch(event) {
    event.preventDefault();
    const trimmed = question.trim();

    if (trimmed.length < 5) {
      setNotice('Please enter a research question with at least 5 characters.');
      return;
    }

    if (submitting || ['queued', 'running'].includes(currentRun?.status)) {
      setNotice('A research run is already in progress. Please wait for it to finish.');
      return;
    }

    setSubmitting(true);
    setNotice('');

    try {
      const data = await apiRequest('/api/research', {
        method: 'POST',
        body: JSON.stringify({ question: trimmed }),
      });

      const run = {
        id: data.id,
        question: trimmed,
        status: data.status || 'queued',
        stage: 'Queued',
        logs: [],
        papers: [],
        report_files: [],
        error: null,
      };

      setCurrentRun(run);
      setQuestion('');
      setNotice('Your question was submitted successfully.');
      setActive('research');
      await refreshData();
    } catch (error) {
      setNotice(`Could not start research: ${error.message}`);
    } finally {
      setSubmitting(false);
    }
  }

  function navigate(id) {
    setActive(id);
    setNotice('');
    setMobileNav(false);
    setFilter('');
    if (id === 'papers' || id === 'reports') refreshData();
  }

  const filteredPapers = papers.filter(p =>
    `${p.title || ''} ${p.authors || ''} ${p.year || ''} ${p.question || ''}`
      .toLowerCase().includes(filter.toLowerCase())
  );

  const filteredReports = reports.filter(r =>
    `${r.filename || ''} ${r.question || ''} ${r.run_id || ''}`
      .toLowerCase().includes(filter.toLowerCase())
  );

  const status = currentRun?.status;
  const isRunning = status === 'queued' || status === 'running';

  function reportTitle(filename) {
    return {
      'report.md': 'Research report (Markdown)',
      'report.docx': 'Research report (Word)',
      'evaluation.json': 'Research evaluation (JSON)',
    }[filename] || filename;
  }

  return <div className="app-shell">
    {mobileNav && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
      <div className="brand-row">
        <div className="brand-mark"><BookOpen size={20} strokeWidth={2.1}/></div>
        <div><div className="brand-name">AutoResearch</div><div className="brand-caption">ACADEMIC WORKSPACE</div></div>
        <button className="icon-button mobile-close" aria-label="Close menu" onClick={() => setMobileNav(false)}><X size={18}/></button>
      </div>

      <div className="nav-label">WORKSPACE</div>
      <nav className="main-nav" aria-label="Main navigation">
        {navItems.map(({ id, label, icon: Icon }) =>
          <button key={id} className={`nav-item ${active === id ? 'active' : ''}`} onClick={() => navigate(id)}>
            <Icon size={18}/>
            <span>{label}</span>
            {id === 'papers' && <span className="nav-count">{papers.length}</span>}
          </button>
        )}
      </nav>

      <div className="sidebar-bottom">
        <div className="help-card">
          <div className="help-icon"><CircleHelp size={17}/></div>
          <div><strong>Getting started</strong><p>Submit a question and follow your research pipeline.</p></div>
        </div>
        <button className="nav-item settings-item" onClick={() => setNotice('The API currently uses your local Python environment and configuration.')}><Settings2 size={18}/><span>Settings</span></button>
        <div className="profile-row">
          <div className="avatar">AR</div>
          <div className="profile-copy"><strong>Research workspace</strong><span>{apiOnline ? 'Backend connected' : 'Backend offline'}</span></div>
          <span className="status-dot" title={apiOnline ? 'Backend online' : 'Backend offline'}/>
        </div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}><Menu size={20}/></button>
        <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{navItems.find(n => n.id === active)?.label}</strong></div>
        <div className="topbar-right">
          <span className="preview-pill"><span/> {apiOnline ? 'Backend connected' : 'Backend disconnected'}</span>
        </div>
      </header>

      <div className="page-content">
        {active === 'research' && <section className="research-page">
          <div className="eyebrow"><Sparkles size={15}/> YOUR RESEARCH WORKSPACE</div>
          <h1>What are you researching<br className="desktop-break"/> today?</h1>
          <p className="page-intro">Turn a research question into a structured literature review, with papers, evidence, and a report in one place.</p>

          <form className="question-card" onSubmit={startResearch}>
            <label htmlFor="research-question">Research question</label>
            <textarea id="research-question" value={question} onChange={e => { setQuestion(e.target.value); setNotice(''); }} placeholder="e.g. How does AI improve cybersecurity threat detection, and what are its limitations?" rows={4}/>
            <div className="question-footer">
              <span className="input-hint"><Clock3 size={14}/> A research run may take a few minutes</span>
              <button className="primary-button" type="submit" disabled={submitting || isRunning || !apiOnline}>
                {submitting || isRunning ? <LoaderCircle size={16} className="spin-icon"/> : <ArrowUpRight size={16}/>}
                {submitting ? 'Submitting…' : isRunning ? 'Research running…' : 'Start research'}
              </button>
            </div>
          </form>

          {!apiOnline && <div className="notice notice-info" role="status">Backend is not reachable. Start the FastAPI server at 127.0.0.1:8000 and refresh this page.</div>}
          {notice && <div className={`notice ${notice.toLowerCase().includes('failed') || notice.toLowerCase().includes('could not') ? 'notice-error' : 'notice-info'}`} role="status">{notice}</div>}

          {currentRun && <section className="run-panel">
            <div className="section-heading">
              <div><h2>Research progress</h2><p>{currentRun.question}</p></div>
              <span className={`run-status status-${currentRun.status}`}>{currentRun.status}</span>
            </div>
            <div className="run-stage">
              {isRunning ? <LoaderCircle size={18} className="spin-icon"/> : status === 'completed' ? <CheckCircle2 size={18}/> : status === 'failed' ? <AlertCircle size={18}/> : <Clock3 size={18}/>}
              <strong>{currentRun.stage || 'Waiting to start'}</strong>
            </div>
            {currentRun.error && <p className="run-error">{currentRun.error}</p>}
            <div className="log-heading"><strong>Pipeline logs</strong><span>{currentRun.logs?.length || 0} lines</span></div>
            <pre className="log-output">{currentRun.logs?.length ? currentRun.logs.join('\n') : 'Waiting for pipeline output…'}</pre>
            {status === 'completed' && <div className="run-downloads">
              {(currentRun.report_files || []).map(filename =>
                <a key={filename} className="secondary-button" href={`${API}/api/reports/${currentRun.id}/${encodeURIComponent(filename)}`} target="_blank" rel="noreferrer">
                  <Download size={15}/> {filename}
                </a>
              )}
              <button className="secondary-button" onClick={() => navigate('reports')}>View all reports</button>
            </div>}
          </section>}

          <div className="section-heading"><div><h2>How it works</h2><p>Your question passes through the existing Python research pipeline.</p></div></div>
          <div className="workflow-grid">
            <div className="workflow-card"><div className="step-number">01</div><div className="workflow-icon lilac"><Search size={19}/></div><h3>Explore papers</h3><p>Retrieve relevant research and organize paper metadata.</p></div>
            <div className="workflow-card"><div className="step-number">02</div><div className="workflow-icon blue"><BookOpen size={19}/></div><h3>Synthesize evidence</h3><p>Analyze themes, verify claims, and identify research gaps.</p></div>
            <div className="workflow-card"><div className="step-number">03</div><div className="workflow-icon green"><FileText size={19}/></div><h3>Build your report</h3><p>Generate and download the literature review and evaluation.</p></div>
          </div>
        </section>}

        {active === 'papers' && <section className="listing-page">
          <div className="page-title-row"><div><div className="eyebrow">YOUR LIBRARY</div><h1>Papers</h1><p className="page-intro">Papers parsed from completed research runs.</p></div><span className="soft-count">{filteredPapers.length} papers</span></div>
          <div className="search-field"><Search size={17}/><input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Search papers by title or question..." aria-label="Search papers"/></div>
          <div className="listing-actions"><span>{apiOnline ? 'Live data from AutoResearch' : 'Backend unavailable'}</span><button className="secondary-button" onClick={refreshData} disabled={refreshing}><RefreshCw size={15}/> Refresh</button></div>
          <div className="paper-list">
            {filteredPapers.map((paper, i) => <article className="paper-card" key={`${paper.run_id || 'paper'}-${paper.title}-${i}`}>
              <div className="paper-icon"><FileText size={20}/></div>
              <div className="paper-main">
                <div className="paper-meta"><span className="tag">Retrieved paper</span><span>{paper.year || 'Year unavailable'}</span></div>
                <h3>{paper.title}</h3>
                <p className="paper-authors">{paper.authors || paper.question || 'From a completed research run'}</p>
                {paper.abstract && <p className="paper-abstract">{paper.abstract}</p>}
              </div>
            </article>)}
            {filteredPapers.length === 0 && <div className="empty-state">{apiOnline ? 'No parsed papers are available yet. Complete a research run first.' : 'Start the backend to load your papers.'}</div>}
          </div>
        </section>}

        {active === 'reports' && <section className="listing-page">
          <div className="page-title-row"><div><div className="eyebrow">OUTPUTS</div><h1>Reports</h1><p className="page-intro">Download reports and evaluation files produced by your research runs.</p></div><span className="soft-count">{filteredReports.length} files</span></div>
          <div className="search-field"><Search size={17}/><input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Search reports or research questions..." aria-label="Search reports"/></div>
          <div className="listing-actions"><span>{apiOnline ? 'Live data from AutoResearch' : 'Backend unavailable'}</span><button className="secondary-button" onClick={refreshData} disabled={refreshing}><RefreshCw size={15}/> Refresh</button></div>
          <div className="report-list">
            {filteredReports.map((report, i) => <article className="report-card" key={`${report.run_id}-${report.filename}-${i}`}>
              <div className="report-file-icon"><FileText size={21}/></div>
              <div className="report-copy">
                <h3>{reportTitle(report.filename)}</h3>
                <p>{report.question || 'Generated by AutoResearch'}</p>
                <div className="report-meta"><span><Clock3 size={13}/>{new Date(report.created_at).toLocaleString()}</span><span>{report.filename}</span></div>
              </div>
              <a className="secondary-button" href={`${API}${report.url}`} target="_blank" rel="noreferrer"><Download size={15}/> Download</a>
            </article>)}
            {filteredReports.length === 0 && <div className="empty-state">{apiOnline ? 'No reports yet. Start a research run to generate files.' : 'Start the backend to load your reports.'}</div>}
          </div>
        </section>}
      </div>

      <footer className="footer"><span>AutoResearch <span className="footer-dot">·</span> Academic workspace</span><span>{apiOnline ? 'Connected to Python API' : 'Waiting for Python API'}</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
