'use client';

import { useEffect, useMemo } from 'react';
import { Alert, Box, Stack, Typography } from '@mui/material';
import { ConfigPanel } from '@/components/ConfigPanel';
import { LogPanel } from '@/components/LogPanel';
import { SessionPanel } from '@/components/SessionPanel';
import { SystemOverviewCard } from '@/components/SystemOverviewCard';
import { TaskProgressPanel } from '@/components/TaskProgressPanel';
import { useStopTaskMutation, useSystemQuery, useTaskDetailQuery, useTasksQuery } from '@/lib/api/hooks';
import { useUiStore } from '@/store/useUiStore';

export default function HomePage() {
  const activeTaskId = useUiStore((s) => s.activeTaskId);
  const hydrate = useUiStore((s) => s.hydrate);
  const hydrated = useUiStore((s) => s.hydrated);
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);

  const system = useSystemQuery();
  const tasks = useTasksQuery();
  const detail = useTaskDetailQuery(activeTaskId || null);
  const stopTask = useStopTaskMutation();

  useEffect(() => {
    if (!hydrated) hydrate();
  }, [hydrate, hydrated]);

  useEffect(() => {
    if (!tasks.data?.length) return;
    if (activeTaskId) {
      const exists = tasks.data.some((t) => t.job_id === activeTaskId);
      if (!exists) setActiveTaskId(tasks.data[0].job_id);
      return;
    }
    setActiveTaskId(tasks.data[0].job_id);
  }, [tasks.data, activeTaskId, setActiveTaskId]);

  const activeSummary = useMemo(
    () => tasks.data?.find((t) => t.job_id === activeTaskId),
    [tasks.data, activeTaskId],
  );

  const activeStatus = detail.data?.status || activeSummary?.status || '';
  const canStopTask = ['queued', 'running', 'resuming', 'recoverable'].includes(String(activeStatus).toLowerCase());

  const handleStopTask = async () => {
    if (!activeTaskId) return;
    await stopTask.mutateAsync(activeTaskId);
  };

  return (
    <Box sx={{ maxWidth: 1720, mx: 'auto', px: { xs: 1.5, md: 3 }, py: { xs: 1.5, md: 2.5 } }}>
      <Stack spacing={2.25}>
        <Stack spacing={0.75}>
          <Typography variant="h4" fontWeight={700}>Sherpa 控制台</Typography>
          <Typography variant="body2" color="text.secondary">
            面向运行中的 fuzz 任务。当前页优先展示阶段、候选、异常信号和详细日志。
          </Typography>
        </Stack>

        <SystemOverviewCard
          data={system.data}
          error={system.isError ? (system.error as Error).message : undefined}
        />

        <Stack direction={{ xs: 'column', xl: 'row' }} spacing={2} alignItems="stretch">
          <Box sx={{ width: { xs: '100%', xl: 380 }, flexShrink: 0 }}>
            <Stack spacing={2}>
              <ConfigPanel />
              <SessionPanel tasks={tasks.data || []} />
            </Stack>
          </Box>

          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Stack spacing={2}>
              {tasks.isError ? <Alert severity="warning">任务列表加载失败</Alert> : null}
              {activeSummary?.error ? <Alert severity="error">{activeSummary.error}</Alert> : null}
              {stopTask.isError ? (
                <Alert severity="error">停止任务失败：{(stopTask.error as Error).message}</Alert>
              ) : null}

              <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} alignItems="stretch">
                <Box sx={{ width: { xs: '100%', lg: 420 }, flexShrink: 0 }}>
                  <TaskProgressPanel
                    detail={detail.data}
                    onStopTask={handleStopTask}
                    stopDisabled={!activeTaskId || !canStopTask}
                    stopLoading={stopTask.isPending}
                  />
                </Box>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <LogPanel detail={detail.data} />
                </Box>
              </Stack>
            </Stack>
          </Box>
        </Stack>
      </Stack>
    </Box>
  );
}
