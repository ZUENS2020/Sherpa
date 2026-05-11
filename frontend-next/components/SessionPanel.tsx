'use client';

import { Card, CardContent, Chip, FormControl, InputLabel, MenuItem, Select, Stack, Typography } from '@mui/material';
import type { TaskSummary } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';

function shortId(id: string): string {
  return id.slice(0, 8);
}

export function SessionPanel({ tasks }: { tasks: TaskSummary[] }) {
  const activeTaskId = useUiStore((s) => s.activeTaskId);
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);
  const activeTask = tasks.find((task) => task.job_id === activeTaskId) || tasks[0];

  return (
    <Card
      variant="outlined"
      sx={{
        background: 'var(--tianheng-surface)',
        borderColor: 'rgba(15, 23, 42, 0.08)',
      }}
    >
      <CardContent>
        <Stack spacing={1.5}>
          <Typography variant="h6">会话绑定</Typography>
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            <Chip size="small" variant="outlined" label={`任务 ${tasks.length}`} />
            {activeTask ? (
              <Chip size="small" color={activeTask.status === 'ERROR' ? 'error' : activeTask.status === 'RUNNING' ? 'warning' : 'default'} label={`${activeTask.status} · ${activeTask.repo || 'batch'}`} />
            ) : null}
            {activeTask ? (
              <Chip size="small" variant="outlined" label={`Frontier ${Number(activeTask.fuzz_coverage_frontier_summary?.top_input_count || 0)}/${Number(activeTask.fuzz_coverage_frontier_summary?.top_frontier_function_count || 0)}`} />
            ) : null}
            {activeTask ? (
              <Chip size="small" variant="outlined" label={`Replay ${Number(activeTask.fuzz_coverage_replay_processed_inputs || 0)}/${Number(activeTask.fuzz_coverage_replay_total_inputs || 0)}`} />
            ) : null}
          </Stack>
          <FormControl fullWidth size="small">
            <InputLabel id="session-select-label">选择任务</InputLabel>
            <Select
              labelId="session-select-label"
              label="选择任务"
              value={activeTaskId}
              onChange={(e) => setActiveTaskId(String(e.target.value || ''))}
            >
              {tasks.map((task) => (
                <MenuItem key={task.job_id} value={task.job_id}>
                  #{shortId(task.job_id)} | {task.status} | {task.repo || 'batch'} | frontier={Number(task.fuzz_coverage_frontier_summary?.top_input_count || 0)} | replay={Number(task.fuzz_coverage_replay_processed_inputs || 0)}/{Number(task.fuzz_coverage_replay_total_inputs || 0)} | crash={task.crash_vuln_candidate_count || 0}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
      </CardContent>
    </Card>
  );
}
