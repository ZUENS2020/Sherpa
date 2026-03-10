'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControlLabel,
  Stack,
  Switch,
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

function parseBudgetSeconds(input: string, fallback: number, unlimited: boolean): number {
  if (unlimited) return 0;
  return toPositiveInt(input, fallback);
}

function toNonNegativeInt(input: string, fallback: number): number {
  const num = Number.parseInt(input, 10);
  return Number.isFinite(num) && num >= 0 ? num : fallback;
}

function toRangeInt(input: string, fallback: number, min: number, max: number): number {
  const num = Number.parseInt(input, 10);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, num));
}

export function ConfigPanel() {
  const cfgQuery = useConfigQuery();
  const saveCfg = useSaveConfigMutation();
  const submitTask = useSubmitTaskMutation();
  const setActiveTaskId = useUiStore((s) => s.setActiveTaskId);

  const [repoUrl, setRepoUrl] = useState('');
  const [totalBudget, setTotalBudget] = useState('900');
  const [runBudget, setRunBudget] = useState('900');
  const [totalBudgetUnlimited, setTotalBudgetUnlimited] = useState(false);
  const [runBudgetUnlimited, setRunBudgetUnlimited] = useState(false);
  const [maxTokens, setMaxTokens] = useState('1000');
  const [coverageLoopMaxRounds, setCoverageLoopMaxRounds] = useState('3');
  const [maxFixRounds, setMaxFixRounds] = useState('3');
  const [sameErrorMaxRetries, setSameErrorMaxRetries] = useState('1');
  const [unlimitedRoundBudget, setUnlimitedRoundBudget] = useState('7200');

  const [statusText, setStatusText] = useState('');
  const [statusType, setStatusType] = useState<'success' | 'error' | 'info'>('info');

  useEffect(() => {
    if (!cfgQuery.data) return;
    const configuredBudget = Number(cfgQuery.data.fuzz_time_budget);
    const isUnlimitedBudget = Number.isFinite(configuredBudget) && configuredBudget <= 0;
    const normalizedBudget = !isUnlimitedBudget && Number.isFinite(configuredBudget) && configuredBudget > 0
      ? Math.floor(configuredBudget)
      : 900;
    setTotalBudget(String(normalizedBudget));
    setRunBudget(String(normalizedBudget));
    setTotalBudgetUnlimited(isUnlimitedBudget);
    setRunBudgetUnlimited(isUnlimitedBudget);
    const configuredUnlimitedRoundBudget = Number(cfgQuery.data.sherpa_run_unlimited_round_budget_sec);
    const normalizedUnlimitedRoundBudget =
      Number.isFinite(configuredUnlimitedRoundBudget) && configuredUnlimitedRoundBudget >= 0
        ? Math.floor(configuredUnlimitedRoundBudget)
        : 7200;
    setUnlimitedRoundBudget(String(normalizedUnlimitedRoundBudget));
    const normalizedMaxFixRounds =
      Number.isFinite(Number(cfgQuery.data.max_fix_rounds)) && Number(cfgQuery.data.max_fix_rounds) >= 0
        ? Math.floor(Number(cfgQuery.data.max_fix_rounds))
        : 3;
    const normalizedSameErrorMaxRetries =
      Number.isFinite(Number(cfgQuery.data.same_error_max_retries)) && Number(cfgQuery.data.same_error_max_retries) >= 0
        ? Math.floor(Number(cfgQuery.data.same_error_max_retries))
        : 1;
    setMaxFixRounds(String(normalizedMaxFixRounds));
    setSameErrorMaxRetries(String(normalizedSameErrorMaxRetries));
  }, [cfgQuery.data]);

  const mergedConfig = useMemo<WebConfig | null>(() => {
    if (!cfgQuery.data) return null;
    const total = parseBudgetSeconds(totalBudget, 900, totalBudgetUnlimited);
    const unlimitedRoundBudgetSec = toNonNegativeInt(unlimitedRoundBudget, 7200);
    const mergedMaxFixRounds = toRangeInt(maxFixRounds, 3, 0, 20);
    const mergedSameErrorMaxRetries = toRangeInt(sameErrorMaxRetries, 1, 0, 10);
    return {
      ...cfgQuery.data,
      fuzz_time_budget: total,
      sherpa_run_unlimited_round_budget_sec: unlimitedRoundBudgetSec,
      max_fix_rounds: mergedMaxFixRounds,
      same_error_max_retries: mergedSameErrorMaxRetries,
      fuzz_use_docker: true,
      fuzz_docker_image: cfgQuery.data.fuzz_docker_image || 'auto',
    };
  }, [cfgQuery.data, totalBudget, totalBudgetUnlimited, unlimitedRoundBudget, maxFixRounds, sameErrorMaxRetries]);

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

    const total = parseBudgetSeconds(totalBudget, 900, totalBudgetUnlimited);
    const runFallback = total > 0 ? total : 900;
    const run = parseBudgetSeconds(runBudget, runFallback, runBudgetUnlimited);
    const tokens = toPositiveInt(maxTokens, 1000);
    const loopRounds = toRangeInt(coverageLoopMaxRounds, 3, 1, 5);
    const taskMaxFixRounds = toRangeInt(maxFixRounds, 3, 0, 20);
    const taskSameErrorMaxRetries = toRangeInt(sameErrorMaxRetries, 1, 0, 10);

    try {
      setStatusType('info');
      setStatusText('正在提交任务...');
      const res = await submitTask.mutateAsync({
        repoUrl: repo,
        totalTimeBudget: total,
        runTimeBudget: run,
        maxTokens: tokens,
        coverageLoopMaxRounds: loopRounds,
        maxFixRounds: taskMaxFixRounds,
        sameErrorMaxRetries: taskSameErrorMaxRetries,
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

          <Stack spacing={1}>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                label="总时长(秒)"
                value={totalBudget}
                onChange={(e) => setTotalBudget(e.target.value)}
                size="small"
                type="number"
                fullWidth
                disabled={totalBudgetUnlimited}
                helperText={totalBudgetUnlimited ? '不限时（0）' : undefined}
              />
              <FormControlLabel
                control={(
                  <Switch
                    checked={totalBudgetUnlimited}
                    onChange={(e) => setTotalBudgetUnlimited(e.target.checked)}
                  />
                )}
                label="不限时"
              />
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                label="单次时长(秒)"
                value={runBudget}
                onChange={(e) => setRunBudget(e.target.value)}
                size="small"
                type="number"
                fullWidth
                disabled={runBudgetUnlimited}
                helperText={runBudgetUnlimited ? '不限时（0）' : undefined}
              />
              <FormControlLabel
                control={(
                  <Switch
                    checked={runBudgetUnlimited}
                    onChange={(e) => setRunBudgetUnlimited(e.target.checked)}
                  />
                )}
                label="不限时"
              />
            </Stack>
          </Stack>

          <TextField
            label="Max Tokens"
            value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value)}
            size="small"
            type="number"
            fullWidth
          />

          <TextField
            label="Coverage 闭环轮数"
            value={coverageLoopMaxRounds}
            onChange={(e) => setCoverageLoopMaxRounds(e.target.value)}
            size="small"
            type="number"
            fullWidth
            helperText="范围 1-5，默认 3"
          />

          <TextField
            label="build/fix 最大轮数"
            value={maxFixRounds}
            onChange={(e) => setMaxFixRounds(e.target.value)}
            size="small"
            type="number"
            fullWidth
            helperText="max_fix_rounds，范围 0-20（0 表示禁用 fix_build 重试）"
          />

          <TextField
            label="同错误重试次数"
            value={sameErrorMaxRetries}
            onChange={(e) => setSameErrorMaxRetries(e.target.value)}
            size="small"
            type="number"
            fullWidth
            helperText="same_error_max_retries，范围 0-10"
          />

          <TextField
            label="不限时时单轮上限(秒)"
            value={unlimitedRoundBudget}
            onChange={(e) => setUnlimitedRoundBudget(e.target.value)}
            size="small"
            type="number"
            fullWidth
            helperText="0 表示完全不限时；建议默认 7200（2小时）"
          />

          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              onClick={handleSave}
              disabled={saveCfg.isPending || !mergedConfig}
            >
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
