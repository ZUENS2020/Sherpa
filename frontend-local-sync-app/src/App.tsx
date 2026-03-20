import { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  ListChecks,
  Zap,
  Search,
  Link as LinkIcon,
  Settings,
  Download,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Plus,
  AlertTriangle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState('tasks');
  const [apiBaseUrl, setApiBaseUrl] = useState(() => {
    return localStorage.getItem('sherpaApiBaseUrl') || 'https://dev.zuens2020.work';
  });
  const [tasksData, setTasksData] = useState<any[]>([]);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [selectedTask, setSelectedTask] = useState<any>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [fetchError, setFetchError] = useState('');
  const [isStartModalOpen, setIsStartModalOpen] = useState(false);
  const [repoUrl, setRepoUrl] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState('');
  const [taskSearch, setTaskSearch] = useState('');
  const [taskStatusFilter, setTaskStatusFilter] = useState<'ALL' | 'RUNNING' | 'SUCCESS' | 'FAILED'>('ALL');
  const [taskToCancel, setTaskToCancel] = useState<any>(null);

  const normalizedTasks = (tasksData || []).map((task: any) => {
    const status = String(task?.status || 'UNKNOWN').toUpperCase();
    const stage = String(task?.active_child_status || task?.stage || 'UNKNOWN').toUpperCase();
    return {
      ...task,
      status,
      stage,
      progress: Number(task?.progress || 0),
      id: task?.job_id || task?.id || 'UNKNOWN',
      repo: task?.repo || 'Unknown Repo',
    };
  });

  const performanceSeries = Array.isArray(systemStatus?.telemetry?.performance_series)
    ? systemStatus.telemetry.performance_series
    : [];
  const agentHealthMatrix = Array.isArray(systemStatus?.telemetry?.agent_health_matrix)
    ? systemStatus.telemetry.agent_health_matrix
    : [];
  const runningTasks = normalizedTasks.filter((t: any) => t.status === 'RUNNING');
  const failedTasks = normalizedTasks.filter((t: any) => t.status === 'FAILED' || t.status === 'ERROR');
  const totalTasks = normalizedTasks.length;
  const filteredTasks = normalizedTasks.filter((task: any) => {
    const status = String(task?.status || '').toUpperCase();
    const stage = String(task?.stage || task?.active_child_status || '').toUpperCase();
    const id = String(task?.id || task?.job_id || '').toUpperCase();
    const repo = String(task?.repo || '').toUpperCase();
    const query = taskSearch.trim().toUpperCase();
    const matchesStatus =
      taskStatusFilter === 'ALL' ||
      (taskStatusFilter === 'RUNNING' && status === 'RUNNING') ||
      (taskStatusFilter === 'SUCCESS' && (status === 'SUCCESS' || status === 'COMPLETED')) ||
      (taskStatusFilter === 'FAILED' && (status === 'FAILED' || status === 'ERROR'));
    const matchesQuery =
      query.length === 0 ||
      id.includes(query) ||
      repo.includes(query) ||
      status.includes(query) ||
      stage.includes(query);
    return matchesStatus && matchesQuery;
  });

  const crashByRepo = failedTasks.reduce((acc: Record<string, number>, task: any) => {
    acc[task.repo] = (acc[task.repo] || 0) + 1;
    return acc;
  }, {});
  const crashDistribution = Object.entries(crashByRepo)
    .map(([repo, count]) => ({ repo, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 4);
  const maxCrashCount = crashDistribution.length ? crashDistribution[0].count : 1;

  // Task Configuration State
  const [totalDuration, setTotalDuration] = useState(() => localStorage.getItem('sherpaTotalDuration') || '900');
  const [isTotalUnlimited, setIsTotalUnlimited] = useState(() => localStorage.getItem('sherpaIsTotalUnlimited') === 'true');
  const [singleDuration, setSingleDuration] = useState(() => localStorage.getItem('sherpaSingleDuration') || '900');
  const [isSingleUnlimited, setIsSingleUnlimited] = useState(() => localStorage.getItem('sherpaIsSingleUnlimited') === 'true');
  const [maxTokens, setMaxTokens] = useState(() => localStorage.getItem('sherpaMaxTokens') || '0');
  const [unlimitedRoundLimit, setUnlimitedRoundLimit] = useState(() => localStorage.getItem('sherpaUnlimitedRoundLimit') || '7200');

  const handleSaveTaskConfig = () => {
    localStorage.setItem('sherpaTotalDuration', totalDuration);
    localStorage.setItem('sherpaIsTotalUnlimited', isTotalUnlimited.toString());
    localStorage.setItem('sherpaSingleDuration', singleDuration);
    localStorage.setItem('sherpaIsSingleUnlimited', isSingleUnlimited.toString());
    localStorage.setItem('sherpaMaxTokens', maxTokens);
    localStorage.setItem('sherpaUnlimitedRoundLimit', unlimitedRoundLimit);
  };

  const handleStartFuzzing = async () => {
    if (!repoUrl) {
      setStartError('Repository URL is required.');
      return;
    }
    setIsStarting(true);
    setStartError('');
    try {
      const resp = await fetch(`${apiBaseUrl}/api/task`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          jobs: [
            {
              code_url: repoUrl,
              total_duration: isTotalUnlimited ? -1 : Number(totalDuration),
              single_duration: isSingleUnlimited ? -1 : Number(singleDuration),
              max_tokens: Number(maxTokens),
              unlimited_round_limit: Number(unlimitedRoundLimit)
            }
          ]
        }),
      });
      if (resp.ok) {
        setIsStartModalOpen(false);
        setRepoUrl('');
      } else {
        setStartError(`Failed to start fuzzing: ${resp.status}`);
      }
    } catch (err) {
      console.error('Failed to start fuzzing:', err);
      setStartError('Error connecting to API.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopTask = async (jobId: string) => {
    try {
      const resp = await fetch(`${apiBaseUrl}/api/task/${jobId}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (resp.ok) {
        // Refresh tasks list
        const tasksResp = await fetch(`${apiBaseUrl}/api/tasks`);
        if (tasksResp.ok) {
          const data = await tasksResp.json();
          if (data.items) {
            setTasksData(data.items);
          }
        }
      }
    } catch (err) {
      console.error('Failed to stop task:', err);
    }
  };

  const handleConfirmCancelTask = async () => {
    if (!taskToCancel) {
      return;
    }
    const jobId = taskToCancel.job_id || taskToCancel.id;
    setTaskToCancel(null);
    await handleStopTask(jobId);
  };

  const handleRetryTask = async (task: any) => {
    try {
      const resp = await fetch(`${apiBaseUrl}/api/task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobs: [
            {
              code_url: task.repo,
              model: 'MiniMax-M2.7-highspeed',
              max_tokens: Number(maxTokens) || 1000,
              time_budget: isTotalUnlimited ? -1 : Number(totalDuration),
              total_time_budget: isTotalUnlimited ? -1 : Number(totalDuration),
              run_time_budget: isSingleUnlimited ? -1 : Number(singleDuration),
            },
          ],
        }),
      });
      if (resp.ok) {
        // Refresh tasks list
        const tasksResp = await fetch(`${apiBaseUrl}/api/tasks`);
        if (tasksResp.ok) {
          const data = await tasksResp.json();
          if (data.items) {
            setTasksData(data.items);
          }
        }
      }
    } catch (err) {
      console.error('Failed to retry task:', err);
    }
  };

  const handleSaveConfig = async () => {
    setIsSaving(true);
    setSaveMessage('');
    try {
      const resp = await fetch(`${apiBaseUrl}/api/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ apiBaseUrl }),
      });
      if (resp.ok) {
        setSaveMessage('Configuration saved successfully.');
      } else {
        setSaveMessage('Failed to save configuration.');
      }
    } catch (err) {
      console.error('Failed to save config:', err);
      setSaveMessage('Error saving configuration.');
    } finally {
      setIsSaving(false);
    }
  };

  useEffect(() => {
    localStorage.setItem('sherpaApiBaseUrl', apiBaseUrl);
  }, [apiBaseUrl]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const tasksResp = await fetch(`${apiBaseUrl}/api/tasks`);
        if (tasksResp.ok) {
          const data = await tasksResp.json();
          if (data.items) {
            setTasksData(data.items);
          }
        } else {
          setFetchError(`Failed to fetch tasks: ${tasksResp.status}`);
        }

        const sysResp = await fetch(`${apiBaseUrl}/api/system`);
        if (sysResp.ok) {
          const sysData = await sysResp.json();
          setSystemStatus(sysData);
          setFetchError(''); // Clear error on success
        } else {
          setFetchError(`Failed to fetch system status: ${sysResp.status}`);
        }
      } catch (err) {
        console.error('Failed to fetch data:', err);
        setFetchError('Error connecting to API. Please check your API Base URL.');
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return (
    <div className="flex min-h-screen bg-zinc-50 font-body text-zinc-900 selection:bg-emerald-200 selection:text-emerald-900">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 z-50 flex h-full w-64 flex-col border-r border-zinc-200 bg-white py-4">
        <div className="mb-6 px-6">
          <div className="flex flex-col">
            <span className="font-headline text-2xl font-black italic tracking-tighter uppercase text-zinc-900">
              SHERPA
            </span>
            <span className="font-label text-[10px] font-bold tracking-tight uppercase text-zinc-400">
              FUZZING ORCHESTRATION
            </span>
          </div>
        </div>

        <nav className="flex flex-1 flex-col space-y-2">
          <div className="w-full overflow-hidden">
            <a
              href="#"
              onClick={(e) => { e.preventDefault(); setActiveTab('overview'); }}
              className={`flex items-center space-x-4 px-6 py-4 transition-all duration-150 ${
                activeTab === 'overview'
                  ? '-skew-x-12 border-l-4 border-emerald-700 bg-zinc-100 text-emerald-700'
                  : 'border-l-4 border-transparent text-zinc-500 opacity-70 hover:border-orange-600 hover:bg-zinc-100'
              }`}
            >
              <div className={`flex items-center space-x-4 ${activeTab === 'overview' ? 'skew-x-12' : ''}`}>
                <LayoutDashboard className="h-5 w-5" />
                <span className="font-label font-bold tracking-tight uppercase">
                  Overview
                </span>
              </div>
            </a>
          </div>

          <div className="w-full overflow-hidden">
            <a
              href="#"
              onClick={(e) => { e.preventDefault(); setActiveTab('tasks'); }}
              className={`flex items-center space-x-4 px-6 py-4 transition-all duration-150 ${
                activeTab === 'tasks'
                  ? '-skew-x-12 border-l-4 border-emerald-700 bg-zinc-100 text-emerald-700'
                  : 'border-l-4 border-transparent text-zinc-500 opacity-70 hover:border-orange-600 hover:bg-zinc-100'
              }`}
            >
              <div className={`flex items-center space-x-4 ${activeTab === 'tasks' ? 'skew-x-12' : ''}`}>
                <ListChecks className="h-5 w-5" />
                <span className="font-label font-bold tracking-tight uppercase">
                  Tasks
                </span>
              </div>
            </a>
          </div>

          <div className="w-full overflow-hidden">
            <a
              href="#"
              onClick={(e) => { e.preventDefault(); setActiveTab('settings'); }}
              className={`flex items-center space-x-4 px-6 py-4 transition-all duration-150 ${
                activeTab === 'settings'
                  ? '-skew-x-12 border-l-4 border-emerald-700 bg-zinc-100 text-emerald-700'
                  : 'border-l-4 border-transparent text-zinc-500 opacity-70 hover:border-orange-600 hover:bg-zinc-100'
              }`}
            >
              <div className={`flex items-center space-x-4 ${activeTab === 'settings' ? 'skew-x-12' : ''}`}>
                <Settings className="h-5 w-5" />
                <span className="font-label font-bold tracking-tight uppercase">
                  Settings
                </span>
              </div>
            </a>
          </div>
        </nav>

        <div className="mt-auto px-6">
          <button 
            onClick={() => setIsStartModalOpen(true)}
            className="sherpa-slash flex w-full items-center justify-center space-x-2 bg-emerald-700 py-4 font-label text-xs font-black tracking-widest text-white transition-colors hover:bg-emerald-800"
          >
            <span>START FUZZING</span>
            <Zap className="h-4 w-4 fill-current" />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="sherpa-grid relative ml-64 flex flex-1 flex-col min-h-screen overflow-x-hidden">
        {fetchError && (
          <div className="bg-rose-100 border-l-4 border-rose-600 p-4 mx-6 mt-2 flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5 text-rose-600" />
            <p className="font-mono text-sm font-bold text-rose-900">{fetchError}</p>
          </div>
        )}

        {activeTab === 'overview' ? (
          <div className="flex-1 overflow-y-auto">
            <header className="flex items-end justify-between px-6 pb-6 pt-6">
              <div>
                <h1 className="font-headline text-5xl font-black italic leading-none tracking-tighter uppercase">
                  System Overview
                </h1>
                <p className="mt-2 font-label text-sm font-medium tracking-[0.2em] text-emerald-700 uppercase">
                  Real-time Fuzzing & Cluster Health
                </p>
              </div>
              <div className="flex gap-4">
                <div className="text-right">
                  <p className="font-label text-[10px] tracking-widest text-zinc-500 uppercase">
                    Avg Fuzz Time
                  </p>
                  <p className="font-headline text-2xl font-black italic leading-none text-orange-600 uppercase">
                    {systemStatus?.overview?.avg_fuzz_time || '--'}
                  </p>
                </div>
                <div className="h-10 w-[1px] bg-zinc-300"></div>
                <div className="text-right">
                  <p className="font-label text-[10px] tracking-widest text-zinc-500 uppercase">
                    Active Agents
                  </p>
                  <p className="font-headline text-2xl font-black italic leading-none text-emerald-700 uppercase">
                    {systemStatus?.overview?.active_agents || '--'}
                  </p>
                </div>
              </div>
            </header>
            <div className="grid grid-cols-12 gap-6 p-6">
              {/* Top Row: Metrics */}
              <section className="col-span-12 grid grid-cols-4 gap-4">
                <div className="border-l-4 border-orange-600 bg-white p-6 shadow-sm">
                  <p className="mb-2 font-label text-xs font-bold uppercase text-zinc-500">
                    Cluster Health
                  </p>
                  <div className="flex items-end justify-between">
                    <h3 className="font-headline text-4xl font-black italic uppercase">
                      {systemStatus?.overview?.cluster_health || '--'}<span className="text-lg">%</span>
                    </h3>
                    <span className="font-headline text-sm font-bold italic text-emerald-700">
                      {systemStatus?.overview?.cluster_health_trend || '--'}
                    </span>
                  </div>
                </div>
                <div className="border-l-4 border-orange-600 bg-white p-6 shadow-sm">
                  <p className="mb-2 font-label text-xs font-bold uppercase text-zinc-500">
                    Crash Triage Rate
                  </p>
                  <div className="flex items-end justify-between">
                    <h3 className="font-headline text-4xl font-black italic uppercase">
                      {systemStatus?.overview?.crash_triage_rate || '--'}<span className="text-lg">/hr</span>
                    </h3>
                    <span className="font-headline text-sm font-bold italic text-emerald-700">
                      {systemStatus?.overview?.crash_triage_rate_trend || '--'}
                    </span>
                  </div>
                </div>
                <div className="border-l-4 border-orange-600 bg-white p-6 shadow-sm">
                  <p className="mb-2 font-label text-xs font-bold uppercase text-zinc-500">
                    Harnesses Synthesized
                  </p>
                  <div className="flex items-end justify-between">
                    <h3 className="font-headline text-4xl font-black italic uppercase">
                      {systemStatus?.overview?.harnesses_synthesized || '--'}<span className="text-lg">k</span>
                    </h3>
                    <span className="font-headline text-sm font-bold italic text-emerald-700">
                      {systemStatus?.overview?.harnesses_synthesized_trend || '--'}
                    </span>
                  </div>
                </div>
                <div className="border-l-4 border-orange-600 bg-white p-6 shadow-sm">
                  <p className="mb-2 font-label text-xs font-bold uppercase text-zinc-500">
                    Avg Coverage
                  </p>
                  <div className="flex items-end justify-between">
                    <h3 className="font-headline text-4xl font-black italic uppercase">
                      {systemStatus?.overview?.avg_coverage || '--'}<span className="text-lg">%</span>
                    </h3>
                    <span className="font-headline text-sm font-bold italic text-emerald-700">
                      {systemStatus?.overview?.avg_coverage_trend || '--'}
                    </span>
                  </div>
                </div>
              </section>

              {/* Second Row */}
              <section className="col-span-7 grid grid-cols-1 gap-6">
                <div className="border border-zinc-200 bg-white p-8">
                  <div className="mb-6 flex items-start justify-between">
                    <div>
                      <h2 className="font-headline text-2xl font-black italic tracking-tighter uppercase">
                        LangGraph Agent Telemetry
                      </h2>
                      <p className="font-label text-xs tracking-widest text-zinc-500 uppercase">
                        FastAPI + OpenCode Integration
                      </p>
                    </div>
                    <div className="bg-emerald-100 px-3 py-1">
                      <span className="font-mono text-[10px] font-bold text-emerald-700 uppercase">
                        {`Status: ${fetchError ? 'Degraded' : 'Nominal'}`}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-8">
                    <div className="space-y-4">
                      <div className="border-l-4 border-emerald-500 bg-zinc-50 p-4">
                        <p className="mb-1 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          LLM Token Usage
                        </p>
                        <div className="flex items-baseline justify-between">
                          <span className="font-mono text-xl font-bold italic uppercase">
                            {systemStatus?.telemetry?.llm_token_usage || '--'}
                          </span>
                          <span className="font-mono text-xs font-bold text-emerald-500 uppercase">
                            {systemStatus?.telemetry?.llm_token_status || '--'}
                          </span>
                        </div>
                      </div>
                      <div className="border-l-4 border-orange-500 bg-zinc-50 p-4">
                        <p className="mb-1 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          K8s Pod Capacity
                        </p>
                        <div className="flex items-baseline justify-between">
                          <span className="font-mono text-xl font-bold italic uppercase">
                            {systemStatus?.telemetry?.k8s_pod_capacity || '--'}
                          </span>
                          <span className="font-mono text-[10px] font-bold text-orange-500 uppercase">
                            {systemStatus?.telemetry?.k8s_pod_status || '--'}
                          </span>
                        </div>
                      </div>
                      <div className="border-l-4 border-emerald-500 bg-zinc-50 p-4">
                        <p className="mb-1 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          FastAPI Gateway
                        </p>
                        <div className="flex items-baseline justify-between">
                          <span className="font-mono text-xl font-bold italic uppercase">
                            {systemStatus?.telemetry?.fastapi_gateway || '--'}
                          </span>
                          <span className="font-mono text-[10px] font-bold text-emerald-500 uppercase">
                            {systemStatus?.telemetry?.fastapi_status || '--'}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <p className="mb-3 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                        Agent Health Matrix
                      </p>
                      <div className="grid grid-cols-8 gap-1.5">
                        {/* Status Grid */}
                        {(agentHealthMatrix.length ? agentHealthMatrix : []).map((value: number, i: number) => {
                          const isWarning = Number(value) === 0;
                          return (
                            <div
                              key={i}
                              className={`h-6 w-6 ${
                                isWarning ? 'bg-orange-500' : 'bg-emerald-500'
                              }`}
                              style={{ opacity: 1 }}
                            ></div>
                          );
                        })}
                      </div>
                      <div className="mt-4 flex justify-between">
                        <div className="flex gap-4">
                          <div className="flex items-center gap-1">
                            <div className="h-2 w-2 bg-emerald-500"></div>
                            <span className="font-mono text-[8px] uppercase">
                              Stable
                            </span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="h-2 w-2 bg-orange-500"></div>
                            <span className="font-mono text-[8px] uppercase">
                              Warning
                            </span>
                          </div>
                        </div>
                        <span className="font-mono text-[8px] text-zinc-500 uppercase">
                          {agentHealthMatrix.length
                            ? `${agentHealthMatrix.filter((v: number) => Number(v) !== 0).length}/${agentHealthMatrix.length} ACTIVE`
                            : 'NO DATA'}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-8 h-48 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={performanceSeries} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" vertical={false} />
                        <XAxis dataKey="time" stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="left" stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="right" orientation="right" stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#18181b', border: 'none', borderRadius: 0, color: '#fff' }}
                          itemStyle={{ fontSize: 12, fontFamily: 'monospace' }}
                          labelStyle={{ color: '#a1a1aa', fontSize: 10, marginBottom: 4 }}
                        />
                        <Line name="Harnesses" yAxisId="left" type="monotone" dataKey="throughput" stroke="#047857" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#047857', stroke: '#fff' }} />
                        <Line name="Crash Rate" yAxisId="right" type="monotone" dataKey="latency" stroke="#ea580c" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#ea580c', stroke: '#fff' }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 flex items-center gap-6">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-emerald-700"></div>
                      <span className="font-mono text-[10px] font-bold text-zinc-500 uppercase">Harnesses (k/s)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-orange-600"></div>
                      <span className="font-mono text-[10px] font-bold text-zinc-500 uppercase">Crash Rate (c/s)</span>
                    </div>
                  </div>
                </div>
              </section>

              {/* Right: Crash Distribution */}
              <section className="col-span-5 bg-zinc-900 p-8 text-white">
                <div className="mb-8 flex items-start justify-between">
                  <div>
                    <h2 className="font-headline text-2xl font-black italic tracking-tighter text-white uppercase">
                      Crash Distribution
                    </h2>
                    <p className="font-label text-xs tracking-widest text-zinc-400 uppercase">
                      Crashes by Target
                    </p>
                  </div>
                  <div className="sherpa-slash bg-orange-600 px-3 py-1">
                    <span className="font-label text-xs font-black italic uppercase">
                      P99 Critical
                    </span>
                  </div>
                </div>
                <div className="space-y-6">
                  {crashDistribution.length ? crashDistribution.map((item, idx) => {
                    const ratio = Math.max(5, Math.round((item.count / maxCrashCount) * 100));
                    const colorClass = idx === 0 ? 'bg-rose-600' : idx === 1 ? 'bg-orange-500' : 'bg-emerald-500';
                    return (
                      <div className="space-y-2" key={`${item.repo}-${idx}`}>
                        <div className="flex justify-between font-mono text-[10px] tracking-widest text-zinc-400 uppercase">
                          <span>{item.repo}</span>
                          <span className="text-white">{item.count}</span>
                        </div>
                        <div className="h-1 bg-zinc-700">
                          <div className={`h-full ${colorClass}`} style={{ width: `${ratio}%` }}></div>
                        </div>
                      </div>
                    );
                  }) : (
                    <div className="font-mono text-xs text-zinc-400 uppercase">No crash distribution data</div>
                  )}
                </div>
                <div className="mt-12 border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center gap-4">
                    <div className="text-orange-500">
                      <AlertTriangle className="h-9 w-9" />
                    </div>
                    <div>
                      <p className="font-label text-xs font-bold italic tracking-tighter text-white uppercase">
                        {failedTasks.length ? 'Failed Tasks Detected' : 'No Active Crash Alert'}
                      </p>
                      <p className="font-mono text-[9px] text-zinc-400 uppercase">
                        {failedTasks.length
                          ? `${failedTasks.length} tasks currently in FAILED/ERROR state.`
                          : 'No failed task signals in current snapshot.'}
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Bottom Row */}
              <section className="col-span-12 grid grid-cols-12 gap-8">
                {/* Tasks Summary */}
                <div className="col-span-4 bg-zinc-100 p-8">
                  <h2 className="mb-6 font-headline text-xl font-black italic tracking-tighter uppercase">
                    Active Tasks
                  </h2>
                  <div className="space-y-1">
                    {runningTasks.slice(0, 3).map((task: any) => (
                      <div key={task.id} className="flex items-center justify-between border-l-4 border-emerald-700 bg-white p-3">
                        <span className="font-label text-xs font-bold tracking-widest uppercase">
                          {task.stage}
                        </span>
                        <span className="bg-emerald-100 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-800 uppercase">
                          {Math.max(0, Math.min(100, Number(task.progress || 0)))}%
                        </span>
                      </div>
                    ))}
                    {!runningTasks.length && (
                      <div className="bg-white p-3 font-mono text-xs text-zinc-500 uppercase">
                        No running tasks
                      </div>
                    )}
                  </div>
                </div>

                {/* Task Execution Summary */}
                <div
                  className="relative col-span-8 border border-zinc-200 bg-white p-8"
                  style={{
                    background:
                      'linear-gradient(135deg, transparent 95%, #2ecc71 5%)',
                  }}
                >
                  <div className="mb-6 flex items-end justify-between">
                    <div>
                      <h2 className="font-headline text-xl font-black italic tracking-tighter uppercase">
                        Task Execution Summary
                      </h2>
                      <p className="font-label text-xs tracking-widest text-zinc-500 uppercase">
                        Regional Operations Metrics
                      </p>
                    </div>
                    <div className="flex gap-4">
                      <div className="text-right">
                        <p className="mb-1 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          Failure Rate
                        </p>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xl font-black italic text-emerald-500">
                            {systemStatus?.execution?.summary?.failure_rate || '--'}
                          </span>
                          <span className="bg-emerald-100 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-600 uppercase">
                            Stable
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 grid h-40 grid-cols-3 gap-8">
                    <div className="flex flex-col justify-between border-l-4 border-emerald-700 bg-zinc-50 p-4">
                      <div>
                        <p className="mb-2 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          Fuzzing Jobs
                        </p>
                        <div className="flex items-baseline gap-2">
                          <span className="font-headline text-4xl font-black italic uppercase">
                            {systemStatus?.execution?.summary?.fuzzing_jobs_24h || '--'}
                          </span>
                          <span className="font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
                            Jobs / 24h
                          </span>
                        </div>
                      </div>
                      <div className="mt-2 h-12 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={performanceSeries}>
                            <Area type="monotone" dataKey="throughput" stroke="#047857" fill="#047857" fillOpacity={0.2} strokeWidth={2} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="flex flex-col justify-between border-l-4 border-orange-500 bg-zinc-50 p-4">
                      <div>
                        <p className="mb-2 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                          Cluster Load
                        </p>
                        <div className="flex items-baseline gap-2">
                          <span className="font-headline text-4xl font-black italic uppercase">
                            {systemStatus?.execution?.summary?.cluster_load_peak || '--'}
                          </span>
                          <span className="font-mono text-[10px] tracking-widest text-orange-500 uppercase">
                            PEAK
                          </span>
                        </div>
                      </div>
                      <div className="mt-2 h-12 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={performanceSeries}>
                            <Area type="monotone" dataKey="latency" stroke="#ea580c" fill="#ea580c" fillOpacity={0.2} strokeWidth={2} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="space-y-2 overflow-hidden border-l-4 border-emerald-700 bg-zinc-50 p-4">
                      <p className="mb-1 font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                        Process Status
                      </p>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 bg-emerald-500"></div>
                        <span className="font-mono text-[9px] tracking-tighter uppercase">
                          {`Tasks API: ${fetchError ? 'ERROR' : 'OK'}`}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 bg-emerald-500"></div>
                        <span className="font-mono text-[9px] tracking-tighter uppercase">
                          {`FastAPI: ${systemStatus?.telemetry?.fastapi_status || '--'}`}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 bg-emerald-500"></div>
                        <span className="font-mono text-[9px] tracking-tighter uppercase">
                          {`Running Jobs: ${runningTasks.length}`}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-6 flex items-center gap-12 border-t border-zinc-200 pt-4">
                    <div>
                      <p className="font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                        Repos Queued
                      </p>
                      <p className="font-mono text-lg font-bold uppercase">
                        {systemStatus?.execution?.summary?.repos_queued || '--'} <span className="text-xs">REPOS</span>
                      </p>
                    </div>
                    <div>
                      <p className="font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                        Avg Triage Time
                      </p>
                      <p className="font-mono text-lg font-bold uppercase">
                        {systemStatus?.execution?.summary?.avg_triage_time_ms || '--'} <span className="text-xs">ms</span>
                      </p>
                    </div>
                    <div>
                      <p className="font-label text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
                        Success Ratio
                      </p>
                      <p className="font-mono text-lg font-bold uppercase">
                        {systemStatus?.execution?.summary?.success_ratio || '--'} <span className="text-xs">%</span>
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        ) : activeTab === 'tasks' ? (
          <div className="flex-1 overflow-y-auto">
            <header className="flex items-end justify-between px-6 pb-6 pt-6">
              <div>
                <h1 className="font-headline text-5xl font-black italic leading-none tracking-tighter uppercase">
                  Tasks
                </h1>
                <p className="mt-2 font-label text-sm font-medium tracking-[0.2em] text-emerald-700 uppercase">
                  Real-time Job Queue & Execution State
                </p>
              </div>
              <div className="flex flex-col items-end gap-4">
                <div className="text-right">
                  <p className="font-label text-[10px] tracking-widest text-zinc-500 uppercase">
                    Total Jobs
                  </p>
                  <p className="font-headline text-2xl font-black italic leading-none text-orange-600 uppercase">
                    {systemStatus?.tasks_tab_metrics?.total_jobs || String(totalTasks)}
                  </p>
                </div>
                <div className="flex items-center gap-6">
                  <div className="relative">
                    <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-400" />
                    <input
                      type="text"
                      value={taskSearch}
                      onChange={(e) => setTaskSearch(e.target.value)}
                      placeholder="SEARCH JOBS..."
                      className="w-72 bg-zinc-100 py-3 pl-12 pr-4 font-mono text-xs tracking-tight outline-none focus:ring-2 focus:ring-emerald-700"
                    />
                  </div>
                  <div className="flex bg-zinc-100 p-1">
                    {(['ALL', 'RUNNING', 'SUCCESS', 'FAILED'] as const).map((filter) => (
                      <button
                        key={filter}
                        onClick={() => setTaskStatusFilter(filter)}
                        className={`px-4 py-2 font-mono text-[10px] font-bold uppercase transition-colors ${
                          taskStatusFilter === filter
                            ? 'bg-emerald-700 text-white'
                            : 'text-zinc-500 hover:text-emerald-700'
                        }`}
                      >
                        {filter}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </header>

            {/* Metrics Row */}
            <div className="grid grid-cols-4 gap-4 px-6">
              <div className="border-l-4 border-emerald-700 bg-white p-6 shadow-sm">
                <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400">
                  Total Jobs
                </p>
                <p className="font-mono text-3xl font-black text-zinc-900">
                  {systemStatus?.tasks_tab_metrics?.total_jobs || String(totalTasks)}
                </p>
              </div>
              <div className="border-l-4 border-orange-600 bg-white p-6 shadow-sm">
                <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400">
                  Execs / Sec
                </p>
                <p className="font-mono text-3xl font-black text-zinc-900">
                  {systemStatus?.tasks_tab_metrics?.execs_per_sec || '--'}<span className="text-xs">K/S</span>
                </p>
              </div>
              <div className="border-l-4 border-emerald-500 bg-white p-6 shadow-sm">
                <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400">
                  Success Rate
                </p>
                <p className="font-mono text-3xl font-black text-zinc-900">
                  {systemStatus?.tasks_tab_metrics?.success_rate || '--'}<span className="text-xs">%</span>
                </p>
              </div>
              <div className="border-l-4 border-rose-600 bg-white p-6 shadow-sm">
                <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400">
                  Failed Tasks
                </p>
                <p className="font-mono text-3xl font-black text-zinc-900">{systemStatus?.tasks_tab_metrics?.failed_tasks || String(failedTasks.length)}</p>
              </div>
            </div>

            {/* Data Table */}
            <div className="mt-6 bg-white shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left">
                <thead>
                  <tr className="bg-zinc-100">
                    <th className="px-6 py-4 font-mono text-[10px] font-black tracking-widest uppercase text-zinc-500">
                      JOB_ID
                    </th>
                    <th className="px-6 py-4 font-mono text-[10px] font-black tracking-widest uppercase text-zinc-500">
                      REPOSITORY_URL
                    </th>
                    <th className="px-6 py-4 font-mono text-[10px] font-black tracking-widest uppercase text-zinc-500">
                      CURRENT_STAGE
                    </th>
                    <th className="px-6 py-4 font-mono text-[10px] font-black tracking-widest uppercase text-zinc-500">
                      STATUS
                    </th>
                    <th className="w-40 px-6 py-4 text-center font-mono text-[10px] font-black tracking-widest uppercase text-zinc-500">
                      ACTIONS
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100">
                  {filteredTasks.map((task: any, index: number) => {
                    const status = task.status;
                    const stage = task.stage;
                    const progress = Math.max(0, Math.min(100, Number(task.progress || 0)));
                    return (
                    <tr
                      key={index}
                      onClick={() => setSelectedTask(task)}
                      className={`group transition-colors hover:bg-zinc-50 cursor-pointer ${
                        status === 'FAILED' || status === 'ERROR'
                          ? 'border-l-4 border-rose-600'
                          : ''
                      }`}
                    >
                      <td className="px-6 py-6">
                        <span
                          className={`font-mono text-xs font-bold ${
                            status === 'FAILED' || status === 'ERROR'
                              ? 'text-rose-600'
                              : 'text-emerald-700'
                          }`}
                        >
                          {task.job_id || task.id}
                        </span>
                      </td>
                      <td className="px-6 py-6">
                        <div className="flex items-center space-x-2">
                          <LinkIcon
                            className={`h-4 w-4 ${
                              status === 'FAILED' || status === 'ERROR'
                                ? 'text-rose-600'
                                : 'text-zinc-400'
                            }`}
                          />
                          <span className="font-mono text-xs text-zinc-600">
                            {task.repo || 'Unknown Repo'}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-6">
                        {status === 'RUNNING' ? (
                          <div className="flex items-center space-x-3">
                            <span className="bg-zinc-100 px-2 py-1 font-mono text-[10px] font-bold uppercase">
                              {stage}
                            </span>
                            <div className="relative h-1 w-12 bg-zinc-100">
                              <div
                                className="absolute inset-0 bg-orange-600"
                                style={{ width: `${progress}%` }}
                              ></div>
                            </div>
                          </div>
                        ) : status === 'FAILED' || status === 'ERROR' ? (
                          <span className="bg-rose-100 px-2 py-1 font-mono text-[10px] font-bold uppercase text-rose-900">
                            {stage}
                          </span>
                        ) : (
                          <span className="bg-zinc-100 px-2 py-1 font-mono text-[10px] font-bold uppercase">
                            {stage}
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-6">
                        {status === 'RUNNING' && (
                          <span className="inline-flex items-center space-x-1 bg-orange-100 px-3 py-1 font-mono text-[10px] font-bold uppercase text-orange-700">
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-600"></span>
                            <span>Running</span>
                          </span>
                        )}
                        {(status === 'SUCCESS' || status === 'COMPLETED') && (
                          <span className="inline-flex items-center space-x-1 bg-emerald-100 px-3 py-1 font-mono text-[10px] font-bold uppercase text-emerald-700">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-600"></span>
                            <span>Completed</span>
                          </span>
                        )}
                        {(status === 'FAILED' || status === 'ERROR') && (
                          <span className="inline-flex items-center space-x-1 bg-rose-100 px-3 py-1 font-mono text-[10px] font-bold uppercase text-rose-700">
                            <span className="h-1.5 w-1.5 rounded-full bg-rose-600"></span>
                            <span>Failed</span>
                          </span>
                        )}
                        {status === 'QUEUED' && (
                          <span className="inline-flex items-center space-x-1 bg-zinc-100 px-3 py-1 font-mono text-[10px] font-bold uppercase text-zinc-700">
                            <span className="h-1.5 w-1.5 rounded-full bg-zinc-400"></span>
                            <span>Queued</span>
                          </span>
                        )}
                      </td>
                      <td className="w-40 px-6 py-6 text-center">
                        {status === 'RUNNING' && (
                          <div className="flex justify-center">
                            <button
                              className="inline-flex items-center gap-2 border border-rose-600 bg-rose-50 px-4 py-2 font-label text-[10px] font-black tracking-[0.3em] uppercase text-rose-700 transition-all hover:bg-rose-100 hover:text-rose-800"
                              onClick={(e) => {
                                e.stopPropagation();
                                setTaskToCancel(task);
                              }}
                              title="Cancel Task"
                            >
                              <AlertTriangle className="h-3.5 w-3.5" />
                              <span>Cancel</span>
                            </button>
                          </div>
                        )}
                        {(status === 'SUCCESS' || status === 'COMPLETED') && (
                          <div className="flex justify-center">
                            <button
                              className="text-zinc-400 transition-colors hover:text-emerald-700"
                              onClick={() => setSelectedTask(task)}
                              title="View Details"
                            >
                              <Download className="h-5 w-5" />
                            </button>
                          </div>
                        )}
                        {(status === 'FAILED' || status === 'ERROR') && (
                          <div className="flex justify-center">
                            <button
                              className="text-rose-600 transition-transform hover:scale-110"
                              onClick={() => handleRetryTask(task)}
                              title="Retry Task"
                            >
                              <RotateCcw className="h-5 w-5" />
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  )})}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          <footer className="flex items-center justify-between border-t border-zinc-200 px-6 pt-8 font-mono text-[10px] tracking-widest uppercase text-zinc-400">
            <div>{`SHOWING 1-${filteredTasks.length} OF ${systemStatus?.tasks_tab_metrics?.total_jobs || totalTasks} JOBS`}</div>
            <div className="flex space-x-4">
              <button className="flex items-center transition-colors hover:text-emerald-700">
                <ChevronLeft className="h-4 w-4" /> PREV
              </button>
              <div className="flex space-x-2">
                <span className="font-bold text-emerald-700">01</span>
              </div>
              <button className="flex items-center transition-colors hover:text-emerald-700">
                NEXT <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </footer>
          </div>
        ) : activeTab === 'settings' ? (
          <div className="flex-1 overflow-y-auto p-6">
            <header className="flex items-end justify-between pt-2 pb-6">
              <div>
                <h1 className="font-headline text-5xl font-black italic leading-none tracking-tighter uppercase">
                  Settings
                </h1>
                <p className="mt-2 font-label text-sm font-medium tracking-[0.2em] text-emerald-700 uppercase">
                  API Configuration & Runtime Options
                </p>
              </div>
            </header>
            <div className="max-w-2xl bg-white p-8 shadow-sm border border-zinc-200">
              <h2 className="font-headline text-2xl font-black italic tracking-tighter uppercase mb-6">
                API Configuration
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block font-label text-xs font-bold tracking-widest uppercase text-zinc-500 mb-2">
                    API Base URL
                  </label>
                  <input
                    type="text"
                    value={apiBaseUrl}
                    onChange={(e) => setApiBaseUrl(e.target.value)}
                    placeholder="https://dev.zuens2020.work"
                    className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700"
                  />
                  <p className="mt-2 font-mono text-[10px] text-zinc-400">
                    The base URL for the Sherpa backend API.
                  </p>
                </div>
                <div className="pt-4">
                  <button
                    onClick={handleSaveConfig}
                    disabled={isSaving}
                    className="bg-emerald-700 text-white px-6 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-emerald-800 disabled:opacity-50"
                  >
                    {isSaving ? 'Saving...' : 'Save Configuration'}
                  </button>
                  {saveMessage && (
                    <p className={`mt-2 font-mono text-xs ${saveMessage.includes('Error') || saveMessage.includes('Failed') ? 'text-rose-600' : 'text-emerald-700'}`}>
                      {saveMessage}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : null}

      {/* Cancel Task Modal */}
      {taskToCancel && (
        <div
          className="fixed inset-0 z-[105] flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm p-4"
          onClick={() => setTaskToCancel(null)}
        >
          <div
            className="w-full max-w-lg bg-white shadow-2xl border border-zinc-200 flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-zinc-200 p-6 bg-zinc-50">
              <div>
                <h3 className="font-headline text-2xl font-black italic tracking-tighter uppercase text-zinc-900">
                  Cancel Task
                </h3>
                <p className="font-mono text-xs text-zinc-500 mt-1">
                  {taskToCancel.job_id || taskToCancel.id}
                </p>
              </div>
              <button
                onClick={() => setTaskToCancel(null)}
                className="text-zinc-400 hover:text-zinc-900 transition-colors"
              >
                <Plus className="h-6 w-6 rotate-45" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <p className="font-label text-xs font-bold tracking-widest uppercase text-zinc-500">
                {taskToCancel.repo || 'Unknown Repo'}
              </p>
              <p className="font-mono text-sm text-zinc-700">
                Cancel this running task now? This cannot be undone.
              </p>
            </div>
            <div className="border-t border-zinc-200 p-6 bg-zinc-50 flex justify-end gap-3">
              <button
                onClick={() => setTaskToCancel(null)}
                className="bg-white border border-zinc-300 text-zinc-700 px-5 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-zinc-100"
              >
                Keep Running
              </button>
              <button
                onClick={handleConfirmCancelTask}
                className="bg-rose-600 text-white px-5 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-rose-700"
              >
                Cancel Task
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Decorative Background Elements */}
      <div className="fixed bottom-0 right-0 -z-10 h-96 w-96 translate-x-32 translate-y-32 skew-x-[45deg] bg-emerald-700/5"></div>
      <div className="fixed bottom-24 right-24 -z-10 h-32 w-1 -skew-x-[45deg] bg-orange-600/20"></div>
      <div className="fixed bottom-32 right-32 -z-10 h-32 w-1 -skew-x-[45deg] bg-emerald-700/20"></div>
      </main>

      {/* Start Fuzzing Modal */}
      {isStartModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm p-4">
          <div className="w-full max-w-xl bg-white shadow-2xl border border-zinc-200 flex flex-col">
            <div className="flex items-center justify-between border-b border-zinc-200 p-6 bg-zinc-50">
              <div>
                <h3 className="font-headline text-2xl font-black italic tracking-tighter uppercase text-zinc-900">
                  Initialize Fuzzing
                </h3>
                <p className="font-mono text-xs text-zinc-500 mt-1">
                  CONFIGURE NEW FUZZING TASK
                </p>
              </div>
              <button
                onClick={() => setIsStartModalOpen(false)}
                className="text-zinc-400 hover:text-zinc-900 transition-colors"
              >
                <Plus className="h-6 w-6 rotate-45" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <label className="block font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mb-2">
                  Target Repository URL
                </label>
                <input
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  placeholder="https://github.com/user/repo"
                  className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700"
                />
              </div>

              <div className="grid grid-cols-1 gap-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1 mr-4">
                    <label className="block font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mb-2">
                      Total Duration (Seconds)
                    </label>
                    <input
                      type="number"
                      value={totalDuration}
                      onChange={(e) => setTotalDuration(e.target.value)}
                      disabled={isTotalUnlimited}
                      className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700 disabled:opacity-50"
                    />
                  </div>
                  <div className="flex flex-col items-center justify-center mt-6">
                    <button
                      type="button"
                      onClick={() => setIsTotalUnlimited(!isTotalUnlimited)}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${isTotalUnlimited ? 'bg-emerald-700' : 'bg-zinc-300'}`}
                    >
                      <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${isTotalUnlimited ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                    <span className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mt-1">Unlimited</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex-1 mr-4">
                    <label className="block font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mb-2">
                      Single Duration (Seconds)
                    </label>
                    <input
                      type="number"
                      value={singleDuration}
                      onChange={(e) => setSingleDuration(e.target.value)}
                      disabled={isSingleUnlimited}
                      className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700 disabled:opacity-50"
                    />
                  </div>
                  <div className="flex flex-col items-center justify-center mt-6">
                    <button
                      type="button"
                      onClick={() => setIsSingleUnlimited(!isSingleUnlimited)}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${isSingleUnlimited ? 'bg-emerald-700' : 'bg-zinc-300'}`}
                    >
                      <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${isSingleUnlimited ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                    <span className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mt-1">Unlimited</span>
                  </div>
                </div>

                <div>
                  <label className="block font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mb-2">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(e.target.value)}
                    className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700"
                  />
                </div>

                <div>
                  <label className="block font-label text-[10px] font-bold tracking-widest uppercase text-zinc-500 mb-2">
                    Unlimited Time Single Round Limit (Seconds)
                  </label>
                  <input
                    type="number"
                    value={unlimitedRoundLimit}
                    onChange={(e) => setUnlimitedRoundLimit(e.target.value)}
                    className="w-full bg-zinc-100 py-3 px-4 font-mono text-sm tracking-tight outline-none focus:ring-2 focus:ring-emerald-700"
                  />
                  <p className="mt-2 font-mono text-[10px] text-zinc-400">
                    0 means completely unlimited; default 7200 (2 hours) recommended.
                  </p>
                </div>
              </div>
              
              {startError && (
                <div className="bg-rose-100 border-l-4 border-rose-600 p-3 flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4 text-rose-600" />
                  <p className="font-mono text-xs font-bold text-rose-900">{startError}</p>
                </div>
              )}
            </div>
            <div className="border-t border-zinc-200 p-6 bg-zinc-50 flex justify-start space-x-4">
              <button
                onClick={() => {
                  handleSaveTaskConfig();
                  // Optional: show a quick visual feedback or just save silently
                }}
                className="bg-white border border-zinc-300 text-zinc-700 px-6 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-zinc-100"
              >
                Save Config
              </button>
              <button
                onClick={handleStartFuzzing}
                disabled={isStarting}
                className="bg-emerald-700 text-white px-6 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-emerald-800 disabled:opacity-50 flex items-center space-x-2"
              >
                <span>{isStarting ? 'INITIALIZING...' : 'Submit Task'}</span>
                {!isStarting && <Zap className="h-3 w-3 fill-current" />}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Task Detail Modal */}
      {selectedTask && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm p-4">
          <div className="w-full max-w-3xl bg-white shadow-2xl border border-zinc-200 flex flex-col max-h-[90vh]">
            <div className="flex items-center justify-between border-b border-zinc-200 p-6 bg-zinc-50">
              <div>
                <h3 className="font-headline text-2xl font-black italic tracking-tighter uppercase text-zinc-900">
                  Task Details
                </h3>
                <p className="font-mono text-xs text-zinc-500 mt-1">
                  {selectedTask.job_id || selectedTask.id}
                </p>
              </div>
              <button
                onClick={() => setSelectedTask(null)}
                className="text-zinc-400 hover:text-zinc-900 transition-colors"
              >
                <Plus className="h-6 w-6 rotate-45" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400 mb-1">Repository</p>
                  <p className="font-mono text-sm text-zinc-900">{selectedTask.repo || 'Unknown'}</p>
                </div>
                <div>
                  <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400 mb-1">Status</p>
                  <p className={`font-mono text-sm font-bold ${
                    (selectedTask.status || '').toUpperCase() === 'FAILED' || (selectedTask.status || '').toUpperCase() === 'ERROR'
                      ? 'text-rose-600'
                      : (selectedTask.status || '').toUpperCase() === 'RUNNING'
                      ? 'text-orange-600'
                      : 'text-emerald-700'
                  }`}>
                    {(selectedTask.status || 'UNKNOWN').toUpperCase()}
                  </p>
                </div>
                <div>
                  <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400 mb-1">Stage</p>
                  <p className="font-mono text-sm text-zinc-900">{selectedTask.active_child_status || selectedTask.stage || 'UNKNOWN'}</p>
                </div>
                <div>
                  <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400 mb-1">Progress</p>
                  <p className="font-mono text-sm text-zinc-900">{selectedTask.progress || 0}%</p>
                </div>
              </div>
              
              <div>
                <p className="font-label text-[10px] font-bold tracking-widest uppercase text-zinc-400 mb-2">Raw Data</p>
                <pre className="bg-zinc-900 text-zinc-100 p-4 rounded-md font-mono text-xs overflow-x-auto">
                  {JSON.stringify(selectedTask, null, 2)}
                </pre>
              </div>
            </div>
            <div className="border-t border-zinc-200 p-6 bg-zinc-50 flex justify-end">
              <button
                onClick={() => setSelectedTask(null)}
                className="bg-zinc-900 text-white px-6 py-2 font-label text-xs font-bold tracking-widest uppercase transition-colors hover:bg-zinc-800"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
