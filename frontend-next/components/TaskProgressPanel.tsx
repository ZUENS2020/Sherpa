'use client';

import React from 'react';
import { Alert, Box, Button, Card, CardContent, Chip, LinearProgress, Stack, Typography } from '@mui/material';
import type { TaskDetail } from '@/lib/api/schemas';

type CrashVulnCandidate = {
  validation_status?: string;
  target_api?: string;
  target_name?: string;
  crash_type?: string;
  classification?: string;
  confidence?: number;
};

function statusColor(status: string): 'default' | 'warning' | 'success' | 'error' | 'info' {
  if (status === 'success') return 'success';
  if (status === 'error') return 'error';
  if (status === 'running') return 'warning';
  if (status === 'queued') return 'info';
  return 'default';
}

function vulnStatusColor(status: string): 'default' | 'warning' | 'success' | 'error' | 'info' {
  if (status === 'real_bug') return 'error';
  if (status === 'likely_bug') return 'warning';
  if (status === 'false_positive') return 'success';
  if (status === 'inconclusive') return 'info';
  return 'default';
}

interface TaskProgressPanelProps {
  detail?: TaskDetail;
  onStopTask?: () => void;
  stopDisabled?: boolean;
  stopLoading?: boolean;
}

export function TaskProgressPanel({ detail, onStopTask, stopDisabled = true, stopLoading = false }: TaskProgressPanelProps) {
  const c = detail?.children_status;
  const total = c?.total || 0;
  const finished = (c?.success || 0) + (c?.error || 0);
  const percent = total > 0 ? Math.round((finished / total) * 100) : 0;
  const activeChild = detail?.children?.find((x) => x.status === 'running') || detail?.children?.[0];
  const activeResult = activeChild?.result && typeof activeChild.result === 'object'
    ? (activeChild.result as Record<string, unknown>)
    : null;
  const fixRounds = activeResult
    ? `${Number(activeResult.fix_build_attempts || 0)}/${Number(activeResult.max_fix_rounds || 0)}`
    : '';
  const errorSig = activeResult
    ? String(activeResult.build_error_signature_after || activeResult.build_error_signature_before || '')
    : '';
  const failFastReason = activeResult
    ? String(activeResult.fix_build_terminal_reason || '')
    : '';
  const activeCandidate = ((
    activeChild?.latest_crash_vuln_candidate && Object.keys(activeChild.latest_crash_vuln_candidate).length > 0
      ? activeChild.latest_crash_vuln_candidate
      : detail?.latest_crash_vuln_candidate
  ) || {}) as CrashVulnCandidate;
  const crashVulnCount = Number(activeChild?.crash_vuln_candidate_count || detail?.crash_vuln_candidate_count || 0);
  const analysisVulnCount = Number(activeChild?.vuln_candidate_count || detail?.vuln_candidate_count || 0);
  const vulnStatus = String(activeCandidate.validation_status || '');
  const vulnTarget = String(activeCandidate.target_api || activeCandidate.target_name || '');
  const vulnType = String(activeCandidate.crash_type || activeCandidate.classification || '');
  const stageText = activeResult
    ? String(activeResult.last_step || activeResult.coverage_improve_mode || activeResult.crash_analysis_verdict || '')
    : '';
  const coverageRound = activeResult
    ? `${Number(activeResult.coverage_loop_round || 0)}/${Number(activeResult.coverage_loop_max_rounds || 0)}`
    : '';
  const targetApi = activeResult
    ? String(activeResult.coverage_target_api || activeResult.selected_target_api || activeResult.synthesize_selected_target_api || '')
    : '';
  const replayBinaryCount = Number(activeChild?.fuzz_coverage_replay_binary_count || detail?.fuzz_coverage_replay_binary_count || 0);
  const replayPending = Number(activeChild?.fuzz_coverage_replay_pending_inputs || detail?.fuzz_coverage_replay_pending_inputs || 0);
  const replayFailed = Number(activeChild?.fuzz_coverage_replay_failed_inputs || detail?.fuzz_coverage_replay_failed_inputs || 0);
  const replayProcessed = Number(activeChild?.fuzz_coverage_replay_processed_inputs || detail?.fuzz_coverage_replay_processed_inputs || 0);
  const replayTotal = Number(activeChild?.fuzz_coverage_replay_total_inputs || detail?.fuzz_coverage_replay_total_inputs || 0);
  const replayReady = Boolean(activeChild?.fuzz_coverage_replay_stage_success || detail?.fuzz_coverage_replay_stage_success);
  const frontierInputCount = Number(activeChild?.fuzz_coverage_frontier_summary?.top_input_count || detail?.fuzz_coverage_frontier_summary?.top_input_count || 0);
  const frontierFunctionCount = Number(activeChild?.fuzz_coverage_frontier_summary?.top_frontier_function_count || detail?.fuzz_coverage_frontier_summary?.top_frontier_function_count || 0);
  const runFeedback = (activeChild?.fuzz_coverage_run_feedback_summary
    || detail?.fuzz_coverage_run_feedback_summary
    || {}) as Record<string, unknown>;
  const functionGapCount = Number(runFeedback['function_gap_count'] || 0);
  const pathFrontierCount = Number(runFeedback['path_frontier_count'] || 0);
  const topFunctionGaps = Array.isArray(runFeedback['top_function_gaps'])
    ? (runFeedback['top_function_gaps'] as Array<Record<string, unknown>>).slice(0, 3)
    : [];
  const topPathFrontiers = Array.isArray(runFeedback['top_path_frontiers'])
    ? (runFeedback['top_path_frontiers'] as Array<Record<string, unknown>>).slice(0, 2)
    : [];

  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        background: 'var(--tianheng-surface)',
        borderColor: 'rgba(15, 23, 42, 0.08)',
      }}
    >
      <CardContent sx={{ height: '100%', overflow: 'auto' }}>
        <Stack spacing={1.25}>
          <Stack direction={{ xs: 'column', md: 'row' }} alignItems={{ xs: 'flex-start', md: 'center' }} justifyContent="space-between" spacing={1}>
            <Stack spacing={0.5}>
              <Typography variant="h6">任务进度</Typography>
              <Typography variant="body2" color="text.secondary">
                跟踪当前主任务的阶段、修复节奏和漏洞候选状态。
              </Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip size="small" color={statusColor(detail?.status || 'unknown')} label={detail?.status || 'unknown'} />
              <Button
                variant="outlined"
                color="error"
                size="small"
                onClick={onStopTask}
                disabled={stopDisabled || stopLoading}
              >
                {stopLoading ? '停止中...' : '停止任务'}
              </Button>
            </Stack>
          </Stack>

          {detail?.error ? <Alert severity="error">{detail.error}</Alert> : null}

          <Typography variant="body2" color="text.secondary">
            子任务：{finished}/{total}（running={c?.running || 0}, success={c?.success || 0}, error={c?.error || 0}）
          </Typography>
          <LinearProgress variant="determinate" value={percent} />

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr 1fr', md: 'repeat(4, minmax(0, 1fr))' },
              gap: 1,
            }}
          >
            {[
              ['阶段', stageText || 'n/a'],
              ['Coverage', coverageRound || 'n/a'],
              ['目标 API', targetApi || 'n/a'],
              ['Fix Rounds', fixRounds || '0/0'],
              ['Replay', `${replayBinaryCount} bins · ${replayProcessed}/${replayTotal}`],
              ['Replay Queue', `${replayPending}/${replayFailed}`],
              ['Frontier', `${frontierInputCount} inputs · ${frontierFunctionCount} fns`],
              ['Run Feedback', `${functionGapCount} gaps · ${pathFrontierCount} paths`],
            ].map(([label, value]) => (
              <Box
                key={label}
                sx={{
                  p: 1,
                  borderRadius: '4px',
                  border: '1px solid var(--tianheng-ink)',
                  backgroundColor: 'rgba(255, 250, 240, 0.88)',
                }}
              >
                <Typography variant="caption" color="text.secondary">{label}</Typography>
                <Typography variant="body2" sx={{ mt: 0.5, wordBreak: 'break-word' }}>{value}</Typography>
              </Box>
            ))}
          </Box>

          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            <Chip
              size="small"
              variant="outlined"
              color={detail?.vuln_hunting_enabled || activeChild?.vuln_hunting_enabled ? 'warning' : 'default'}
              label={`漏洞导向：${detail?.vuln_hunting_enabled || activeChild?.vuln_hunting_enabled ? '开启' : '未开启'}`}
            />
            <Chip size="small" variant="outlined" label={`分析候选：${analysisVulnCount}`} />
            <Chip size="small" variant="outlined" color={crashVulnCount > 0 ? 'warning' : 'default'} label={`Crash 候选：${crashVulnCount}`} />
            <Chip size="small" variant="outlined" color={replayReady ? 'success' : 'default'} label={`Replay：${replayBinaryCount}`} />
            <Chip size="small" variant="outlined" label={`Frontier：${frontierInputCount}/${frontierFunctionCount}`} />
            <Chip size="small" variant="outlined" label={`Run Feedback：${functionGapCount}/${pathFrontierCount}`} />
            {vulnStatus ? (
              <Chip size="small" color={vulnStatusColor(vulnStatus)} label={vulnStatus} />
            ) : null}
          </Stack>

          <Typography variant="subtitle2">当前活跃子任务</Typography>
          {activeChild ? (
            <Stack spacing={1}>
              <Alert severity={activeChild.status === 'error' ? 'error' : 'info'}>
                #{activeChild.job_id.slice(0, 8)} | {activeChild.status} | {activeChild.repo || 'unknown'}
              </Alert>
              {fixRounds ? (
                <Typography variant="caption" color="text.secondary">
                  build/fix rounds: {fixRounds}
                </Typography>
              ) : null}
              {errorSig ? (
                <Typography variant="caption" color="text.secondary">
                  error signature: {errorSig.slice(0, 16)}
                </Typography>
              ) : null}
              {failFastReason ? (
                <Alert severity="warning">
                  已触发 fail-fast：{failFastReason}
                </Alert>
              ) : null}
              {vulnStatus ? (
                <Alert severity={vulnStatus === 'real_bug' ? 'error' : 'warning'}>
                  漏洞候选：{vulnTarget || 'unknown'} | {vulnType || 'unknown'} | confidence={Number(activeCandidate.confidence || 0).toFixed(2)}
                </Alert>
              ) : null}
              {topFunctionGaps.length ? (
                <Box
                  sx={{
                    p: 1.25,
                    borderRadius: '4px',
                    border: '1px solid var(--tianheng-ink)',
                    backgroundColor: 'rgba(255, 250, 240, 0.78)',
                  }}
                >
                  <Typography variant="subtitle2" sx={{ mb: 0.75 }}>函数缺口</Typography>
                  <Stack spacing={0.5}>
                    {topFunctionGaps.map((item, index) => {
                      const name = String(item.name || '');
                      const file = String(item.file || '');
                      const line = Number(item.line || 0);
                      const kind = String(item.kind || '');
                      const ratio = Number(item.region_coverage_ratio || 0);
                      return (
                        <Typography key={`${name}-${index}`} variant="caption" sx={{ wordBreak: 'break-word' }}>
                          {name || 'unknown'} · {kind || 'n/a'} · {ratio.toFixed(2)} · {file ? `${file}${line > 0 ? `:${line}` : ''}` : 'no-file'}
                        </Typography>
                      );
                    })}
                  </Stack>
                </Box>
              ) : null}
              {topPathFrontiers.length ? (
                <Box
                  sx={{
                    p: 1.25,
                    borderRadius: '4px',
                    border: '1px solid var(--tianheng-ink)',
                    backgroundColor: 'rgba(255, 250, 240, 0.78)',
                  }}
                >
                  <Typography variant="subtitle2" sx={{ mb: 0.75 }}>路径前沿</Typography>
                  <Stack spacing={0.5}>
                    {topPathFrontiers.map((item, index) => {
                      const relpath = String(item.input_relpath || '');
                      const score = Number(item.frontier_score || 0);
                      const fns = Number(item.covered_function_count || 0);
                      const regions = Number(item.covered_region_count || 0);
                      return (
                        <Typography key={`${relpath}-${index}`} variant="caption" sx={{ wordBreak: 'break-word' }}>
                          {relpath || `input-${index + 1}`} · score {score.toFixed(2)} · fn {fns} · regions {regions}
                        </Typography>
                      );
                    })}
                  </Stack>
                </Box>
              ) : null}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">暂无子任务</Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
