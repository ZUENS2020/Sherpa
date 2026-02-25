'use client';

import { Card, CardContent, FormControl, InputLabel, MenuItem, Select, Stack, Typography } from '@mui/material';
import type { TaskSummary } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';

function shortId(id: string): string {
  return id.slice(0, 8);
}

export function SessionPanel({ tasks }: { tasks: TaskSummary[] }) {
  const activeTaskId = useUiStore((s) => s.activeTaskId);
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={1.5}>
          <Typography variant="h6">会话绑定</Typography>
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
                  #{shortId(task.job_id)} | {task.status} | {task.repo || 'batch'}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
      </CardContent>
    </Card>
  );
}
