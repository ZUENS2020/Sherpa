'use client';

import { useEffect, useMemo } from 'react';
import { Alert, Box, Stack, Typography } from '@mui/material';
import { AppShell } from '@/components/AppShell';
import { SystemOverviewCard } from '@/components/SystemOverviewCard';
import { TaskOverviewPanel } from '@/components/TaskOverviewPanel';
import { TaskDetailCard } from '@/components/TaskDetailCard';
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

  const activeStatus = detail.data?.status || '';
  const canStop = ['queued', 'running', 'resuming', 'recoverable'].includes(activeStatus.toLowerCase());

  const activeSummary = useMemo(
    () => tasks.data?.find((t) => t.job_id === activeTaskId),
    [tasks.data, activeTaskId],
  );

  return (
    <AppShell
      dense
      eyebrow="Monitor / Live operations"
      title="TianHeng 监控台"
      description="任务状态与漏洞候选总览。选择左侧任务查看详情。"
      rail={
        <>
          <Typography className="suzuka-kicker">LIVE</Typography>
          <Typography className="suzuka-kicker">READ-ONLY</Typography>
        </>
      }
    >
      <Stack spacing={1.25} sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <SystemOverviewCard
          data={system.data}
          error={system.isError ? (system.error as Error).message : undefined}
        />

        {tasks.isError ? <Alert severity="warning">任务列表加载失败</Alert> : null}
        {activeSummary?.error ? <Alert severity="error">{activeSummary.error}</Alert> : null}
        {stopTask.isError ? (
          <Alert severity="error">停止失败：{(stopTask.error as Error).message}</Alert>
        ) : null}

        <Stack
          direction="row"
          spacing={1.25}
          alignItems="stretch"
          sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}
        >
          {/* Task list — left column */}
          <Box sx={{ width: 340, flexShrink: 0, minHeight: 0, overflow: 'auto' }}>
            <TaskOverviewPanel tasks={tasks.data ?? []} />
          </Box>

          {/* Task detail — right column */}
          <Box sx={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'auto' }}>
            <TaskDetailCard
              detail={detail.data}
              onStopTask={() => activeTaskId && stopTask.mutate(activeTaskId)}
              stopDisabled={!activeTaskId || !canStop}
              stopLoading={stopTask.isPending}
            />
          </Box>
        </Stack>
      </Stack>
    </AppShell>
  );
}
