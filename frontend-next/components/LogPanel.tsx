'use client';

import { useEffect, useMemo, useRef } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import type { TaskDetail } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';
import { filterLogLines } from './logUtils';

export function LogPanel({ detail }: { detail?: TaskDetail }) {
  const logFilter = useUiStore((s) => s.logFilter);
  const logKeyword = useUiStore((s) => s.logKeyword);
  const autoScrollEnabled = useUiStore((s) => s.autoScrollEnabled);
  const setLogFilter = useUiStore((s) => s.setLogFilter);
  const setLogKeyword = useUiStore((s) => s.setLogKeyword);
  const setAutoScrollEnabled = useUiStore((s) => s.setAutoScrollEnabled);

  const logRef = useRef<HTMLDivElement | null>(null);

  const activeChild =
    detail?.children?.find((child) => child.status === 'running') ||
    detail?.children?.find((child) => child.status === 'error') ||
    detail?.children?.[0];

  const filteredLines = useMemo(() => {
    const raw = activeChild?.log || '';
    return filterLogLines(raw, logFilter, logKeyword);
  }, [activeChild?.log, logFilter, logKeyword]);

  useEffect(() => {
    if (!autoScrollEnabled || !logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [filteredLines, autoScrollEnabled]);

  const onScroll: React.UIEventHandler<HTMLDivElement> = (e) => {
    const el = e.currentTarget;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (!nearBottom && autoScrollEnabled) setAutoScrollEnabled(false);
    if (nearBottom && !autoScrollEnabled) setAutoScrollEnabled(true);
  };

  return (
    <Card variant="outlined" sx={{ minHeight: 360 }}>
      <CardContent>
        <Stack spacing={1.5}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">日志</Typography>
            <Stack direction="row" spacing={1}>
              <TextField
                select
                size="small"
                label="级别"
                value={logFilter}
                onChange={(e) => setLogFilter(e.target.value as 'all' | 'warn' | 'error')}
                sx={{ width: 120 }}
              >
                <MenuItem value="all">全部</MenuItem>
                <MenuItem value="warn">Warn+</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </TextField>
              <TextField
                size="small"
                label="关键词"
                value={logKeyword}
                onChange={(e) => setLogKeyword(e.target.value)}
              />
            </Stack>
          </Stack>

          {!autoScrollEnabled ? (
            <Button variant="outlined" size="small" onClick={() => setAutoScrollEnabled(true)}>
              恢复自动滚动到底部
            </Button>
          ) : null}

          <Box
            ref={logRef}
            onScroll={onScroll}
            sx={{
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              backgroundColor: '#0f172a',
              color: '#e2e8f0',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              p: 1.2,
              maxHeight: 420,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              fontSize: 12,
              lineHeight: 1.5,
            }}
          >
            {filteredLines.length ? filteredLines.join('\n') : '暂无日志输出'}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}
