'use client';

import { Alert, Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import type { SystemStatus } from '@/lib/api/schemas';

type CrashVulnCandidate = {
  validation_status?: string;
  target_api?: string;
  target_name?: string;
};

function fmtDuration(sec?: number): string {
  if (!Number.isFinite(sec) || (sec as number) < 0) return '--';
  const s = Math.floor(sec as number);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

export function SystemOverviewCard({ data, error }: { data?: SystemStatus; error?: string }) {
  if (error) {
    return <Alert severity="warning">系统状态读取失败：{error}</Alert>;
  }

  const jobs = data?.jobs;
  const security = data?.security;
  const latestCandidate = (security?.latest_crash_vuln_candidate || {}) as CrashVulnCandidate;
  const latestStatus = String(latestCandidate.validation_status || '');
  const latestTarget = String(latestCandidate.target_api || latestCandidate.target_name || '');

  return (
    <Card
      variant="outlined"
      sx={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.98), rgba(238,244,252,0.96))',
        borderColor: 'rgba(15, 23, 42, 0.08)',
      }}
    >
      <CardContent>
        <Stack spacing={1.5}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Sherpa 任务总览</Typography>
            <Chip
              size="small"
              color={data?.ok ? 'success' : 'warning'}
              label={data?.ok ? '联机' : '离线'}
              variant="outlined"
            />
          </Stack>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
            <Typography variant="body2">总任务：{jobs?.total ?? 0}</Typography>
            <Typography variant="body2">排队：{jobs?.queued ?? 0}</Typography>
            <Typography variant="body2">运行中：{jobs?.running ?? 0}</Typography>
            <Typography variant="body2">成功：{jobs?.success ?? 0}</Typography>
            <Typography variant="body2">失败：{jobs?.error ?? 0}</Typography>
          </Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            <Chip
              size="small"
              variant="outlined"
              color={security?.vuln_hunting_enabled ? 'warning' : 'default'}
              label={`漏洞导向：${security?.vuln_hunting_enabled ? '开启' : '未开启'}`}
            />
            <Chip size="small" variant="outlined" label={`分析候选：${security?.analysis_vuln_candidate_count ?? 0}`} />
            <Chip
              size="small"
              variant="outlined"
              color={(security?.crash_vuln_candidate_count ?? 0) > 0 ? 'warning' : 'default'}
              label={`Crash 候选：${security?.crash_vuln_candidate_count ?? 0}`}
            />
            {latestStatus ? (
              <Chip size="small" color={latestStatus === 'real_bug' ? 'error' : 'warning'} label={`${latestStatus}${latestTarget ? ` | ${latestTarget}` : ''}`} />
            ) : null}
          </Box>
          {latestStatus ? (
            <Alert severity={latestStatus === 'real_bug' ? 'error' : 'warning'}>
              最近的 crash 漏洞候选：{latestTarget || 'unknown'}
            </Alert>
          ) : null}
          <Typography variant="caption" color="text.secondary">
            服务时间：{data?.server_time_iso || '--'} | Uptime：{fmtDuration(data?.uptime_sec)}
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}
