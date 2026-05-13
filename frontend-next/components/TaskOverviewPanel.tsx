'use client';

import { Box, Card, CardContent, Chip, LinearProgress, Stack, Typography } from '@mui/material';
import type { TaskSummary } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';

function statusColor(s: string): 'default' | 'warning' | 'success' | 'error' | 'info' {
  const v = s.toLowerCase();
  if (v === 'success') return 'success';
  if (v === 'error') return 'error';
  if (v === 'running') return 'warning';
  if (v === 'queued') return 'info';
  return 'default';
}

function shortRepo(repo: string | null | undefined): string {
  if (!repo) return '—';
  return repo.replace(/^https?:\/\/(github\.com|gitlab\.com)\//, '').replace(/\.git$/, '');
}

function shortId(id: string): string {
  return id.slice(0, 8);
}

interface TaskRowProps {
  task: TaskSummary;
  active: boolean;
  onClick: () => void;
}

function TaskRow({ task, active, onClick }: TaskRowProps) {
  const c = task.children_status;
  const total = c?.total ?? 0;
  const finished = (c?.success ?? 0) + (c?.error ?? 0);
  const pct = total > 0 ? Math.round((finished / total) * 100) : 0;
  const coveragePct = Number(
    (task.fuzz_coverage_run_feedback_summary as Record<string, unknown> | undefined)?.['coverage_pct'] ?? 0
  );
  const crashCount = task.crash_vuln_candidate_count ?? 0;
  const running = c?.running ?? 0;

  return (
    <Box
      onClick={onClick}
      sx={{
        p: 1.5,
        borderRadius: '4px',
        border: active ? '2px solid var(--tianheng-green)' : '1px solid var(--tianheng-ink)',
        backgroundColor: active ? 'rgba(0,105,65,0.06)' : 'rgba(255,250,240,0.88)',
        cursor: 'pointer',
        transition: 'all 120ms ease',
        '&:hover': { backgroundColor: 'rgba(0,105,65,0.04)' },
      }}
    >
      <Stack spacing={0.75}>
        {/* repo 名称 */}
        <Typography
          variant="body2"
          sx={{ fontWeight: 600, wordBreak: 'break-word', lineHeight: 1.3 }}
        >
          {shortRepo(task.repo)}
        </Typography>

        {/* chips + 右侧统计：左右两端，chips 区域可换行，统计文字固定右对齐 */}
        <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={1}>
          <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap" alignItems="center" sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              variant="caption"
              sx={{ fontFamily: 'monospace', color: 'var(--tianheng-muted)', flexShrink: 0 }}
            >
              #{shortId(task.job_id)}
            </Typography>
            <Chip size="small" color={statusColor(task.status)} label={task.status.toUpperCase()} />
            {crashCount > 0 ? (
              <Chip size="small" color="error" label={`${crashCount} crash`} />
            ) : null}
          </Stack>

          <Stack spacing={0} alignItems="flex-end" sx={{ flexShrink: 0, pt: '2px' }}>
            <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
              {finished}/{total} 子任务{running > 0 ? ` · ${running} 运行中` : ''}
            </Typography>
            {coveragePct > 0 ? (
              <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                覆盖率 {coveragePct.toFixed(1)}%
              </Typography>
            ) : null}
          </Stack>
        </Stack>

        {total > 0 ? (
          <LinearProgress variant="determinate" value={pct} />
        ) : null}
      </Stack>
    </Box>
  );
}

export function TaskOverviewPanel({ tasks }: { tasks: TaskSummary[] }) {
  const activeTaskId = useUiStore((s) => s.activeTaskId);
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);

  if (!tasks.length) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="body2" color="text.secondary">暂无任务记录</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ height: '100%', overflow: 'auto' }}>
        <Stack spacing={1.25}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">任务总览</Typography>
            <Chip size="small" variant="outlined" label={`${tasks.length} 个任务`} />
          </Stack>
          {tasks.map((t) => (
            <TaskRow
              key={t.job_id}
              task={t}
              active={t.job_id === activeTaskId}
              onClick={() => setActiveTaskId(t.job_id)}
            />
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
}
