'use client';

import { Alert, Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import type { SystemStatus } from '@/lib/api/schemas';

function StatBox({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <Box
      sx={{
        p: 1.5,
        borderRadius: '4px',
        border: '1px solid var(--tianheng-ink)',
        backgroundColor: accent ? 'var(--tianheng-green)' : 'rgba(255, 250, 240, 0.88)',
        color: accent ? '#f5f0e6' : 'inherit',
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        minWidth: 80,
      }}
    >
      <Typography
        variant="caption"
        sx={{ color: accent ? 'rgba(245,240,230,0.75)' : 'var(--tianheng-muted)', letterSpacing: '0.06em', textTransform: 'uppercase', fontSize: 10 }}
      >
        {label}
      </Typography>
      <Typography
        variant="h6"
        sx={{
          fontFamily: '"Space Grotesk", sans-serif',
          fontWeight: 900,
          lineHeight: 1,
          color: accent ? '#f5f0e6' : 'var(--tianheng-ink)',
        }}
      >
        {value}
      </Typography>
    </Box>
  );
}

export function SystemOverviewCard({ data, error }: { data?: SystemStatus; error?: string }) {
  if (error) {
    return <Alert severity="warning">系统状态读取失败：{error}</Alert>;
  }

  const jobs = data?.jobs;
  const security = data?.security;
  const running = jobs?.running ?? 0;
  const success = jobs?.success ?? 0;
  const errCount = jobs?.error ?? 0;
  const total = jobs?.total ?? 0;
  const crashCount = security?.crash_vuln_candidate_count ?? 0;

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="h6">系统概览</Typography>
            <Chip
              size="small"
              color={data?.ok ? 'success' : 'warning'}
              label={data?.ok ? '在线' : '离线'}
              variant="outlined"
            />
          </Stack>
          {security?.vuln_hunting_enabled ? (
            <Chip size="small" color="warning" label="漏洞导向已开启" />
          ) : null}
        </Stack>

        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 1,
          }}
        >
          <StatBox label="总任务" value={total} />
          <StatBox label="运行中" value={running} accent={running > 0} />
          <StatBox label="已完成" value={success} />
          <StatBox label="失败" value={errCount} />
          <StatBox label="分析候选" value={security?.analysis_vuln_candidate_count ?? 0} />
          <StatBox label="Crash 候选" value={crashCount} accent={crashCount > 0} />
        </Box>
      </CardContent>
    </Card>
  );
}
