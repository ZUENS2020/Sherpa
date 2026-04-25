'use client';

import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Divider,
  MenuItem,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import type { TaskDetail } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';
import { detectLevel, filterLogLines } from './logUtils';

type ChildResult = Record<string, unknown>;
type SignalSection = {
  title: string;
  rows: Array<[string, string]>;
};
type SignalRow = [string, string];

function statusTone(status: string): 'default' | 'warning' | 'success' | 'error' | 'info' {
  if (status === 'success') return 'success';
  if (status === 'error') return 'error';
  if (status === 'running') return 'warning';
  if (status === 'queued') return 'info';
  return 'default';
}

function lineColor(level: 'info' | 'warn' | 'error'): string {
  if (level === 'error') return '#fca5a5';
  if (level === 'warn') return '#fcd34d';
  return '#cbd5e1';
}

function hasSignalValue(row: SignalRow): boolean {
  const [, value] = row;
  return Boolean(value);
}

function hasVisibleSignalValue(row: SignalRow): boolean {
  const [, value] = row;
  return Boolean(value) && value !== '0/0' && value !== '0';
}

export function LogPanel({ detail }: { detail?: TaskDetail }) {
  const logFilter = useUiStore((s) => s.logFilter);
  const logKeyword = useUiStore((s) => s.logKeyword);
  const autoScrollEnabled = useUiStore((s) => s.autoScrollEnabled);
  const setLogFilter = useUiStore((s) => s.setLogFilter);
  const setLogKeyword = useUiStore((s) => s.setLogKeyword);
  const setAutoScrollEnabled = useUiStore((s) => s.setAutoScrollEnabled);

  const logRef = useRef<HTMLDivElement | null>(null);
  const [selectedChildId, setSelectedChildId] = useState('');
  const [viewMode, setViewMode] = useState<'log' | 'signals'>('log');
  const deferredKeyword = useDeferredValue(logKeyword);

  const children = useMemo(() => detail?.children ?? [], [detail?.children]);
  const activeChild =
    children.find((child) => child.status === 'running')
    || children.find((child) => child.status === 'error')
    || children[0];
  const selectedChild =
    children.find((child) => child.job_id === selectedChildId)
    || activeChild;
  const selectedResult = selectedChild?.result && typeof selectedChild.result === 'object'
    ? (selectedChild.result as ChildResult)
    : null;

  useEffect(() => {
    if (!children.length) {
      if (selectedChildId) setSelectedChildId('');
      return;
    }
    if (!selectedChildId || !children.some((child) => child.job_id === selectedChildId)) {
      setSelectedChildId(activeChild?.job_id || children[0].job_id);
    }
  }, [activeChild?.job_id, children, selectedChildId]);

  const filteredLines = useMemo(() => {
    const raw = selectedChild?.log || '';
    return filterLogLines(raw, logFilter, deferredKeyword);
  }, [deferredKeyword, logFilter, selectedChild?.log]);

  const logStats = useMemo(() => {
    const lines = (selectedChild?.log || '').split(/\r?\n/).filter((line) => line.trim());
    let warn = 0;
    let error = 0;
    for (const line of lines) {
      const level = detectLevel(line);
      if (level === 'warn') warn += 1;
      if (level === 'error') error += 1;
    }
    return { total: lines.length, warn, error };
  }, [selectedChild?.log]);

  const signalRows = useMemo(() => {
    const rows: SignalRow[] = [
      ['阶段', String(selectedResult?.last_step || detail?.status || 'unknown')],
      ['漏洞导向', selectedChild?.vuln_hunting_enabled ? 'enabled' : 'disabled'],
      ['分析候选', String(selectedChild?.vuln_candidate_count || 0)],
      ['Crash 候选', String(selectedChild?.crash_vuln_candidate_count || 0)],
      ['Coverage round', `${Number(selectedResult?.coverage_loop_round || 0)}/${Number(selectedResult?.coverage_loop_max_rounds || 0)}`],
      ['Seed profile', String(selectedResult?.coverage_seed_profile || '')],
      ['Error signature', String(selectedResult?.build_error_signature_after || selectedResult?.build_error_signature_before || '')],
      ['Fail-fast', String(selectedResult?.fix_build_terminal_reason || '')],
      ['Crash verdict', String(selectedResult?.crash_analysis_verdict || selectedResult?.crash_triage_label || '')],
      ['Vuln target', String(selectedChild?.latest_crash_vuln_candidate?.target_api || selectedChild?.latest_crash_vuln_candidate?.target_name || '')],
    ];
    return rows.filter(hasVisibleSignalValue);
  }, [detail?.status, selectedChild?.crash_vuln_candidate_count, selectedChild?.latest_crash_vuln_candidate, selectedChild?.vuln_candidate_count, selectedChild?.vuln_hunting_enabled, selectedResult]);

  const signalSections = useMemo<SignalSection[]>(() => {
    const coverageRowsSource: SignalRow[] = [
      ['阶段', String(selectedResult?.last_step || detail?.status || 'unknown')],
      ['Coverage round', `${Number(selectedResult?.coverage_loop_round || 0)}/${Number(selectedResult?.coverage_loop_max_rounds || 0)}`],
      ['Plateau streak', String(selectedResult?.coverage_plateau_streak || 0)],
      ['Seed profile', String(selectedResult?.coverage_seed_profile || '')],
      ['Improve mode', String(selectedResult?.coverage_improve_mode || '')],
      ['Bottleneck', String(selectedResult?.coverage_bottleneck_kind || '')],
    ];
    const coverageRows = coverageRowsSource.filter(hasVisibleSignalValue);

    const crashRowsSource: SignalRow[] = [
      ['Crash triage', String(selectedResult?.crash_triage_label || '')],
      ['Crash verdict', String(selectedResult?.crash_analysis_verdict || '')],
      ['Crash reason', String(selectedResult?.crash_analysis_reason || selectedResult?.crash_triage_reason || '')],
      ['Candidate status', String(selectedChild?.latest_crash_vuln_candidate?.validation_status || '')],
      ['Crash type', String(selectedChild?.latest_crash_vuln_candidate?.crash_type || '')],
      ['Sanitizer', String(selectedChild?.latest_crash_vuln_candidate?.sanitizer || '')],
      ['Target', String(selectedChild?.latest_crash_vuln_candidate?.target_api || selectedChild?.latest_crash_vuln_candidate?.target_name || '')],
    ];
    const crashRows = crashRowsSource.filter(hasSignalValue);

    const repairRowsSource: SignalRow[] = [
      ['Fix rounds', `${Number(selectedResult?.fix_build_attempts || 0)}/${Number(selectedResult?.max_fix_rounds || 0)}`],
      ['Fail-fast', String(selectedResult?.fix_build_terminal_reason || '')],
      ['Error signature', String(selectedResult?.build_error_signature_after || selectedResult?.build_error_signature_before || '')],
      ['Repair mode', String(selectedResult?.repair_mode ? 'enabled' : '')],
      ['Repair origin', String(selectedResult?.repair_origin_stage || '')],
      ['Repair code', String(selectedResult?.repair_error_code || '')],
    ];
    const repairRows = repairRowsSource.filter(hasVisibleSignalValue);

    const securityRowsSource: SignalRow[] = [
      ['漏洞导向', selectedChild?.vuln_hunting_enabled ? 'enabled' : 'disabled'],
      ['分析候选', String(selectedChild?.vuln_candidate_count || 0)],
      ['Crash 候选', String(selectedChild?.crash_vuln_candidate_count || 0)],
      ['候选路径', String(selectedChild?.vuln_candidates_path || '')],
      ['报告路径', String(selectedChild?.crash_vuln_report_path || '')],
      ['复现状态', String(selectedChild?.latest_crash_vuln_candidate?.reproduction_status || '')],
    ];
    const securityRows = securityRowsSource.filter(hasSignalValue);

    return [
      { title: 'Coverage', rows: coverageRows },
      { title: 'Crash', rows: crashRows },
      { title: 'Repair', rows: repairRows },
      { title: 'Security', rows: securityRows },
    ].filter((section) => section.rows.length > 0);
  }, [detail?.status, selectedChild, selectedResult]);

  useEffect(() => {
    if (!autoScrollEnabled || !logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [autoScrollEnabled, filteredLines]);

  const onScroll: React.UIEventHandler<HTMLDivElement> = (e) => {
    const el = e.currentTarget;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (!nearBottom && autoScrollEnabled) setAutoScrollEnabled(false);
    if (nearBottom && !autoScrollEnabled) setAutoScrollEnabled(true);
  };

  return (
    <Card
      variant="outlined"
      sx={{
        minHeight: 520,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,247,252,0.96))',
        borderColor: 'rgba(15, 23, 42, 0.08)',
      }}
    >
      <CardContent>
        <Stack spacing={2}>
          <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', md: 'center' }} spacing={1.5}>
            <Stack spacing={0.5}>
              <Typography variant="h6">运行观察台</Typography>
              <Typography variant="body2" color="text.secondary">
                聚焦当前子任务的实时日志、失败信号和漏洞候选。
              </Typography>
            </Stack>
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
              <Chip size="small" variant="outlined" label={`Lines ${logStats.total}`} />
              <Chip size="small" variant="outlined" color={logStats.warn > 0 ? 'warning' : 'default'} label={`Warn ${logStats.warn}`} />
              <Chip size="small" variant="outlined" color={logStats.error > 0 ? 'error' : 'default'} label={`Error ${logStats.error}`} />
              <Chip
                size="small"
                variant="outlined"
                color={selectedChild?.status === 'running' ? 'warning' : statusTone(selectedChild?.status || '')}
                label={selectedChild ? `${selectedChild.status} · ${selectedChild.job_id.slice(0, 8)}` : 'no-child'}
              />
            </Stack>
          </Stack>

          <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} alignItems="stretch">
            <Stack
              spacing={1.25}
              sx={{
                width: { xs: '100%', lg: 280 },
                flexShrink: 0,
                p: 1.25,
                borderRadius: 2,
                border: '1px solid rgba(15, 23, 42, 0.08)',
                backgroundColor: 'rgba(248, 250, 252, 0.95)',
              }}
            >
              <TextField
                select
                size="small"
                label="子任务"
                value={selectedChild?.job_id || ''}
                onChange={(e) => setSelectedChildId(String(e.target.value || ''))}
                fullWidth
              >
                {children.map((child) => (
                  <MenuItem key={child.job_id} value={child.job_id}>
                    #{child.job_id.slice(0, 8)} | {child.status}
                  </MenuItem>
                ))}
              </TextField>

              <Divider />

              {selectedChild ? (
                <Stack spacing={1}>
                  <Typography variant="subtitle2">任务信号</Typography>
                  {signalRows.length ? (
                    signalRows.map(([label, value]) => (
                      <Box key={label} sx={{ display: 'grid', gridTemplateColumns: '92px 1fr', gap: 1 }}>
                        <Typography variant="caption" color="text.secondary">{label}</Typography>
                        <Typography variant="caption" sx={{ wordBreak: 'break-word' }}>{value}</Typography>
                      </Box>
                    ))
                  ) : (
                    <Typography variant="caption" color="text.secondary">暂无结构化信号</Typography>
                  )}
                </Stack>
              ) : (
                <Alert severity="info">暂无子任务可观察</Alert>
              )}
            </Stack>

            <Stack spacing={1.5} sx={{ flex: 1, minWidth: 0 }}>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
                <Tabs value={viewMode} onChange={(_event, value) => setViewMode(value)} sx={{ minHeight: 40 }}>
                  <Tab value="log" label="实时日志" sx={{ minHeight: 40 }} />
                  <Tab value="signals" label="运行信号" sx={{ minHeight: 40 }} />
                </Tabs>
                <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ ml: { md: 'auto' } }}>
                  <TextField
                    select
                    size="small"
                    label="级别"
                    value={logFilter}
                    onChange={(e) => setLogFilter(e.target.value as 'all' | 'warn' | 'error')}
                    sx={{ width: 120 }}
                  >
                    <MenuItem value="all">全部</MenuItem>
                    <MenuItem value="warn">Warn+</MenuItem>
                    <MenuItem value="error">Error</MenuItem>
                  </TextField>
                  <TextField
                    size="small"
                    label="关键词"
                    value={logKeyword}
                    onChange={(e) => setLogKeyword(e.target.value)}
                  />
                </Stack>
              </Stack>

              {!autoScrollEnabled && viewMode === 'log' ? (
                <Button variant="outlined" size="small" onClick={() => setAutoScrollEnabled(true)} sx={{ alignSelf: 'flex-start' }}>
                  恢复自动滚动到底部
                </Button>
              ) : null}

              {viewMode === 'log' ? (
                <Box
                  ref={logRef}
                  onScroll={onScroll}
                  sx={{
                    border: '1px solid rgba(15, 23, 42, 0.08)',
                    borderRadius: 2,
                    background: 'linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.98))',
                    color: '#e2e8f0',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    maxHeight: 560,
                    overflow: 'auto',
                  }}
                >
                  {filteredLines.length ? (
                    filteredLines.map((line, index) => {
                      const level = detectLevel(line);
                      return (
                        <Box
                          key={`${index}-${line.slice(0, 24)}`}
                          sx={{
                            display: 'grid',
                            gridTemplateColumns: '56px 1fr',
                            gap: 1.5,
                            px: 1.5,
                            py: 0.65,
                            borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
                            backgroundColor:
                              level === 'error'
                                ? 'rgba(127, 29, 29, 0.18)'
                                : level === 'warn'
                                  ? 'rgba(120, 53, 15, 0.16)'
                                  : 'transparent',
                          }}
                        >
                          <Typography variant="caption" sx={{ color: '#64748b', textAlign: 'right', pt: '2px' }}>
                            {String(index + 1).padStart(4, '0')}
                          </Typography>
                          <Typography
                            component="pre"
                            sx={{
                              m: 0,
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                              color: lineColor(level),
                              fontSize: 12,
                              lineHeight: 1.55,
                            }}
                          >
                            {line}
                          </Typography>
                        </Box>
                      );
                    })
                  ) : (
                    <Box sx={{ p: 2 }}>
                      <Typography variant="body2" color="rgba(226, 232, 240, 0.72)">
                        暂无日志输出
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : (
                <Stack
                  spacing={1.5}
                  sx={{
                    p: 1.5,
                    borderRadius: 2,
                    border: '1px solid rgba(15, 23, 42, 0.08)',
                    backgroundColor: 'rgba(248, 250, 252, 0.95)',
                  }}
                >
                  {selectedChild?.error ? <Alert severity="error">{selectedChild.error}</Alert> : null}
                  {selectedChild?.latest_crash_vuln_candidate?.reason ? (
                    <Alert severity={selectedChild.latest_crash_vuln_candidate.validation_status === 'real_bug' ? 'error' : 'warning'}>
                      {selectedChild.latest_crash_vuln_candidate.reason}
                    </Alert>
                  ) : null}
                  <Typography variant="subtitle2">结构化结果</Typography>
                  <Box
                    sx={{
                      display: 'grid',
                      gridTemplateColumns: { xs: '1fr', xl: 'repeat(2, minmax(0, 1fr))' },
                      gap: 1.5,
                    }}
                  >
                    {signalSections.map((section) => (
                      <Box
                        key={section.title}
                        sx={{
                          p: 1.5,
                          borderRadius: 2,
                          border: '1px solid rgba(15, 23, 42, 0.08)',
                          backgroundColor: 'rgba(255, 255, 255, 0.86)',
                        }}
                      >
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>{section.title}</Typography>
                        <Stack spacing={0.75}>
                          {section.rows.map(([label, value]) => (
                            <Box key={`${section.title}-${label}`} sx={{ display: 'grid', gridTemplateColumns: '112px 1fr', gap: 1 }}>
                              <Typography variant="caption" color="text.secondary">{label}</Typography>
                              <Typography variant="caption" sx={{ wordBreak: 'break-word' }}>{value}</Typography>
                            </Box>
                          ))}
                        </Stack>
                      </Box>
                    ))}
                  </Box>
                  <Box
                    sx={{
                      borderRadius: 2,
                      backgroundColor: '#0f172a',
                      color: '#cbd5e1',
                      p: 1.5,
                      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                      fontSize: 12,
                      lineHeight: 1.6,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {JSON.stringify(
                      {
                        result: selectedResult || {},
                        latest_crash_vuln_candidate: selectedChild?.latest_crash_vuln_candidate || {},
                        vuln_candidates_path: selectedChild?.vuln_candidates_path || '',
                        crash_vuln_report_path: selectedChild?.crash_vuln_report_path || '',
                      },
                      null,
                      2,
                    )}
                  </Box>
                </Stack>
              )}
            </Stack>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
