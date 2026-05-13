'use client';

import { Alert, Box, Button, Card, CardContent, Chip, LinearProgress, Stack, Typography } from '@mui/material';
import type { TaskDetail } from '@/lib/api/schemas';

function statusColor(s: string): 'default' | 'warning' | 'success' | 'error' | 'info' {
  const v = s.toLowerCase();
  if (v === 'success') return 'success';
  if (v === 'error') return 'error';
  if (v === 'running' || v === 'resuming') return 'warning';
  if (v === 'queued') return 'info';
  return 'default';
}

function vulnStatusColor(s: string): 'default' | 'warning' | 'success' | 'error' {
  if (s === 'real_bug') return 'error';
  if (s === 'likely_bug') return 'warning';
  if (s === 'false_positive') return 'success';
  return 'default';
}

interface Props {
  detail?: TaskDetail;
  onStopTask?: () => void;
  stopDisabled?: boolean;
  stopLoading?: boolean;
}

export function TaskDetailCard({ detail, onStopTask, stopDisabled = true, stopLoading = false }: Props) {
  const c = detail?.children_status;
  const total = c?.total ?? 0;
  const success = c?.success ?? 0;
  const error = c?.error ?? 0;
  const finished = success + error;
  const running = c?.running ?? 0;
  const pct = total > 0 ? Math.round((finished / total) * 100) : 0;

  const crashCount = detail?.crash_vuln_candidate_count ?? 0;
  const analysisCount = detail?.vuln_candidate_count ?? 0;
  const candidate = detail?.latest_crash_vuln_candidate ?? {};
  const candStatus = String((candidate as Record<string, unknown>).validation_status ?? '');
  const candTarget = String(
    (candidate as Record<string, unknown>).target_api ??
    (candidate as Record<string, unknown>).target_name ?? ''
  );
  const candConfidence = Number((candidate as Record<string, unknown>).confidence ?? 0);

  const runFeedback = detail?.fuzz_coverage_run_feedback_summary as Record<string, unknown> | undefined;
  const coveragePct = Number(runFeedback?.['coverage_pct'] ?? 0);
  const coveredFns = Number(runFeedback?.['covered_functions'] ?? 0);
  const totalFns = Number(runFeedback?.['total_functions'] ?? 0);

  if (!detail) {
    return (
      <Card variant="outlined" sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">请从左侧选择任务</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ height: '100%', overflow: 'auto' }}>
        <Stack spacing={1.5}>
          {/* Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={1}>
            <Stack spacing={0.25}>
              <Typography variant="h6">任务详情</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                {detail.job_id}
              </Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip size="small" color={statusColor(detail.status)} label={detail.status.toUpperCase()} />
              <Button
                variant="outlined"
                color="error"
                size="small"
                onClick={onStopTask}
                disabled={stopDisabled || stopLoading}
              >
                {stopLoading ? '停止中…' : '停止'}
              </Button>
            </Stack>
          </Stack>

          {detail.error ? <Alert severity="error">{detail.error}</Alert> : null}

          {/* Vuln alert */}
          {candStatus ? (
            <Alert severity={candStatus === 'real_bug' ? 'error' : 'warning'}>
              <strong>漏洞候选</strong>：{candTarget || 'unknown'} ·{' '}
              <Chip
                size="small"
                color={vulnStatusColor(candStatus)}
                label={candStatus}
                sx={{ ml: 0.5 }}
              />{' '}
              · 置信度 {(candConfidence * 100).toFixed(0)}%
            </Alert>
          ) : null}

          {/* Progress */}
          <Stack spacing={0.75}>
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                子任务进度 {finished}/{total}
                {running > 0 ? ` · ${running} 运行中` : ''}
              </Typography>
              <Typography variant="body2" color="text.secondary">{pct}%</Typography>
            </Stack>
            <LinearProgress variant="determinate" value={pct} />
          </Stack>

          {/* Stats grid */}
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
              gap: 1,
            }}
          >
            {[
              ['完成', success],
              ['失败', error],
              ['等待', c?.queued ?? 0],
              ['分析候选', analysisCount],
              ['Crash 候选', crashCount],
              ['覆盖率', coveragePct > 0 ? `${coveragePct.toFixed(1)}%` : '—'],
            ].map(([label, value]) => (
              <Box
                key={label}
                sx={{
                  p: 1,
                  borderRadius: '4px',
                  border: '1px solid var(--tianheng-ink)',
                  backgroundColor: 'rgba(255,250,240,0.88)',
                  textAlign: 'center',
                }}
              >
                <Typography variant="caption" color="text.secondary" display="block">
                  {label}
                </Typography>
                <Typography variant="subtitle2" sx={{ mt: 0.25 }}>
                  {value}
                </Typography>
              </Box>
            ))}
          </Box>

          {/* Coverage bar */}
          {totalFns > 0 ? (
            <Stack spacing={0.5}>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="caption" color="text.secondary">函数覆盖</Typography>
                <Typography variant="caption" color="text.secondary">
                  {coveredFns}/{totalFns}
                </Typography>
              </Stack>
              <LinearProgress variant="determinate" value={Math.min((coveredFns / totalFns) * 100, 100)} />
            </Stack>
          ) : null}

          {/* Children list */}
          {detail.children?.length ? (
            <Stack spacing={0.75}>
              <Typography variant="subtitle2">子任务列表</Typography>
              {detail.children.map((child) => (
                <Box
                  key={child.job_id}
                  sx={{
                    px: 1.5,
                    py: 1,
                    borderRadius: '4px',
                    border: '1px solid var(--tianheng-ink)',
                    backgroundColor: 'rgba(255,250,240,0.78)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 1,
                  }}
                >
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'var(--tianheng-muted)' }}>
                    #{child.job_id.slice(0, 8)}
                  </Typography>
                  <Chip size="small" color={statusColor(child.status)} label={child.status} />
                  {Number(child.crash_vuln_candidate_count ?? 0) > 0 ? (
                    <Chip size="small" color="error" label={`${child.crash_vuln_candidate_count} crash`} />
                  ) : null}
                </Box>
              ))}
            </Stack>
          ) : null}
        </Stack>
      </CardContent>
    </Card>
  );
}
