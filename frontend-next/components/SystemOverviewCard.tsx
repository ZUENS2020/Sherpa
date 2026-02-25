'use client';

import { Alert, Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import type { SystemStatus } from '@/lib/api/schemas';

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

  return (
    <Card variant="outlined">
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
          <Typography variant="caption" color="text.secondary">
            服务时间：{data?.server_time_iso || '--'} | Uptime：{fmtDuration(data?.uptime_sec)}
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}
