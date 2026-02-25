'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { useConfigQuery, useSaveConfigMutation, useSubmitTaskMutation } from '@/lib/api/hooks';
import type { WebConfig } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';

function toPositiveInt(input: string, fallback: number): number {
  const num = Number.parseInt(input, 10);
  return Number.isFinite(num) && num > 0 ? num : fallback;
}

export function ConfigPanel() {
  const cfgQuery = useConfigQuery();
  const saveCfg = useSaveConfigMutation();
  const submitTask = useSubmitTaskMutation();
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);

  const [repoUrl, setRepoUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [totalBudget, setTotalBudget] = useState('900');
  const [runBudget, setRunBudget] = useState('900');
  const [maxTokens, setMaxTokens] = useState('1000');
  const [statusText, setStatusText] = useState('');
  const [statusType, setStatusType] = useState<'success' | 'error' | 'info'>('info');

  useEffect(() => {
    if (!cfgQuery.data) return;
    setApiKey(cfgQuery.data.openai_api_key || '');
    setTotalBudget(String(cfgQuery.data.fuzz_time_budget || 900));
    setRunBudget(String(cfgQuery.data.fuzz_time_budget || 900));
  }, [cfgQuery.data]);

  const mergedConfig = useMemo<WebConfig | null>(() => {
    if (!cfgQuery.data) return null;
    const total = toPositiveInt(totalBudget, 900);
    return {
      ...cfgQuery.data,
      openai_api_key: apiKey.trim(),
      fuzz_time_budget: total,
      fuzz_use_docker: true,
      fuzz_docker_image: cfgQuery.data.fuzz_docker_image || 'auto',
    };
  }, [cfgQuery.data, apiKey, totalBudget]);

  const handleSave = async () => {
    if (!mergedConfig) return;
    try {
      setStatusType('info');
      setStatusText('正在保存配置...');
      await saveCfg.mutateAsync(mergedConfig);
      setStatusType('success');
      setStatusText('配置已保存。');
    } catch (e) {
      setStatusType('error');
      setStatusText(e instanceof Error ? e.message : '配置保存失败');
    }
  };

  const handleSubmit = async () => {
    const repo = repoUrl.trim();
    if (!repo) {
      setStatusType('error');
      setStatusText('仓库 URL 不能为空。');
      return;
    }

    const total = toPositiveInt(totalBudget, 900);
    const run = toPositiveInt(runBudget, total);
    const tokens = toPositiveInt(maxTokens, 1000);

    try {
      setStatusType('info');
      setStatusText('正在提交任务...');
      const res = await submitTask.mutateAsync({
        repoUrl: repo,
        totalTimeBudget: total,
        runTimeBudget: run,
        maxTokens: tokens,
      });
      setActiveTaskId(res.job_id);
      setStatusType('success');
      setStatusText(`任务已提交：${res.job_id}`);
    } catch (e) {
      setStatusType('error');
      setStatusText(e instanceof Error ? e.message : '任务提交失败');
    }
  };

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Stack spacing={2}>
          <Typography variant="h6">会话与配置</Typography>

          {cfgQuery.isLoading ? (
            <Box display="flex" justifyContent="center" py={2}><CircularProgress size={20} /></Box>
          ) : null}

          <TextField
            label="仓库 URL"
            placeholder="https://github.com/madler/zlib.git"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            size="small"
            fullWidth
          />
          <TextField
            label="API Key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            size="small"
            fullWidth
          />

          <Stack direction="row" spacing={1}>
            <TextField
              label="总时长(秒)"
              value={totalBudget}
              onChange={(e) => setTotalBudget(e.target.value)}
              size="small"
              type="number"
              fullWidth
            />
            <TextField
              label="单次时长(秒)"
              value={runBudget}
              onChange={(e) => setRunBudget(e.target.value)}
              size="small"
              type="number"
              fullWidth
            />
          </Stack>

          <TextField
            label="Max Tokens"
            value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value)}
            size="small"
            type="number"
            fullWidth
          />

          <Stack direction="row" spacing={1}>
            <Button variant="outlined" onClick={handleSave} disabled={saveCfg.isPending || !mergedConfig}>
              保存配置
            </Button>
            <Button variant="contained" onClick={handleSubmit} disabled={submitTask.isPending}>
              提交任务
            </Button>
          </Stack>

          {statusText ? <Alert severity={statusType}>{statusText}</Alert> : null}
        </Stack>
      </CardContent>
    </Card>
  );
}
