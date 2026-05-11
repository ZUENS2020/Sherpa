'use client';

import { useEffect, useMemo } from 'react';
import { Alert, Box, Stack, Typography } from '@mui/material';
import { AppShell } from '@/components/AppShell';
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
    <AppShell
      dense
      eyebrow="Monitor / Live operations"
      title="TianHeng 监控台"
      description="面向值班和运行观察：只展示任务状态、阶段信号、漏洞候选、replay/frontier 反馈和日志。发起新任务已拆到独立页面，避免监控时误提交。"
      rail={(
        <>
          <Typography className="suzuka-kicker">NO TASK MUTATION</Typography>
          <Typography className="suzuka-kicker">LIVE SIGNALS</Typography>
        </>
      )}
    >
      <Stack spacing={1.25} sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <SystemOverviewCard
          data={system.data}
          error={system.isError ? (system.error as Error).message : undefined}
        />

        <Stack direction="row" spacing={1.25} alignItems="stretch" sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <Box sx={{ width: 330, flexShrink: 0, minHeight: 0, overflow: 'auto', pr: 0.25 }}>
            <SessionPanel tasks={tasks.data || []} />
          </Box>

          <Box sx={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
            <Stack spacing={1.25} sx={{ height: '100%', minHeight: 0, overflow: 'hidden' }}>
              {tasks.isError ? <Alert severity="warning">任务列表加载失败</Alert> : null}
              {activeSummary?.error ? <Alert severity="error">{activeSummary.error}</Alert> : null}
              {stopTask.isError ? (
                <Alert severity="error">停止任务失败：{(stopTask.error as Error).message}</Alert>
              ) : null}

              <Stack direction="row" spacing={1.25} alignItems="stretch" sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
                <Box sx={{ width: 390, flexShrink: 0, minHeight: 0, overflow: 'auto', pr: 0.25 }}>
                  <TaskProgressPanel
                    detail={detail.data}
                    onStopTask={handleStopTask}
                    stopDisabled={!activeTaskId || !canStopTask}
                    stopLoading={stopTask.isPending}
                  />
                </Box>
                <Box sx={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
                  <LogPanel detail={detail.data} />
                </Box>
              </Stack>
            </Stack>
          </Box>
        </Stack>
      </Stack>
    </AppShell>
  );
}
