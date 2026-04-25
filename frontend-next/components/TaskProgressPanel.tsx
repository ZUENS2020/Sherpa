'use client';

import { Alert, Button, Card, CardContent, Chip, LinearProgress, Stack, Typography } from '@mui/material';
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

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={1.5}>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">任务进度</Typography>
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

          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            <Chip
              size="small"
              variant="outlined"
              color={detail?.vuln_hunting_enabled || activeChild?.vuln_hunting_enabled ? 'warning' : 'default'}
              label={`漏洞导向：${detail?.vuln_hunting_enabled || activeChild?.vuln_hunting_enabled ? '开启' : '未开启'}`}
            />
            <Chip size="small" variant="outlined" label={`分析候选：${analysisVulnCount}`} />
            <Chip size="small" variant="outlined" color={crashVulnCount > 0 ? 'warning' : 'default'} label={`Crash 候选：${crashVulnCount}`} />
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
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">暂无子任务</Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
