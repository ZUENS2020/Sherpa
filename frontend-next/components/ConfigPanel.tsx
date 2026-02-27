'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import { getOpencodeProviderModels } from '@/lib/api/client';
import { useConfigQuery, useSaveConfigMutation, useSubmitTaskMutation } from '@/lib/api/hooks';
import type { OpencodeProvider, WebConfig } from '@/lib/api/schemas';
import { useUiStore } from '@/store/useUiStore';

function toPositiveInt(input: string, fallback: number): number {
  const num = Number.parseInt(input, 10);
  return Number.isFinite(num) && num > 0 ? num : fallback;
}

function parseBudgetSeconds(input: string, fallback: number, unlimited: boolean): number {
  if (unlimited) return 0;
  return toPositiveInt(input, fallback);
}

type ProviderType = 'minimax' | 'glm' | 'deepseek' | 'openrouter';

interface ProviderMeta {
  label: string;
  providerId: string;
  baseUrl: string;
  models: string[];
}

interface ProviderState {
  apiKey: string;
  apiKeySet: boolean;
  clearApiKey: boolean;
  model: string;
  customModel: boolean;
  availableModels: string[];
  loadingModels: boolean;
  fetchError: string;
}

const PROVIDER_ORDER: ProviderType[] = ['minimax', 'glm', 'deepseek', 'openrouter'];
const CUSTOM_MODEL_VALUE = '__custom__';

const PROVIDER_META: Record<ProviderType, ProviderMeta> = {
  minimax: {
    label: 'MiniMax',
    providerId: 'minimax',
    baseUrl: 'https://api.minimax.io/v1',
    models: ['minimax/minimax-m2.1', 'minimax/minimax-m2'],
  },
  glm: {
    label: 'GLM (Z.AI)',
    providerId: 'zai',
    baseUrl: 'https://open.bigmodel.cn/api/coding/paas/v4',
    models: ['zai/glm-4.7', 'zai/glm-4.6', 'zai/glm-4.5'],
  },
  deepseek: {
    label: 'DeepSeek',
    providerId: 'deepseek',
    baseUrl: 'https://api.deepseek.com/v1',
    models: ['deepseek/deepseek-reasoner', 'deepseek/deepseek-chat'],
  },
  openrouter: {
    label: 'OpenRouter',
    providerId: 'openrouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    models: ['qwen/qwen3-coder-next', 'deepseek/deepseek-reasoner', 'anthropic/claude-3.7-sonnet'],
  },
};

function providerTypeFromConfigName(raw: string): ProviderType | null {
  const name = (raw || '').trim().toLowerCase();
  if (!name) return null;
  if (name === 'minimax') return 'minimax';
  if (name === 'deepseek') return 'deepseek';
  if (name === 'openrouter') return 'openrouter';
  if (name === 'zai' || name === 'zhipuai' || name === 'glm') return 'glm';
  return null;
}

function makeDefaultProviderState(type: ProviderType): ProviderState {
  const models = [...PROVIDER_META[type].models];
  return {
    apiKey: '',
    apiKeySet: false,
    clearApiKey: false,
    model: models[0] || '',
    customModel: false,
    availableModels: models,
    loadingModels: false,
    fetchError: '',
  };
}

function defaultProviderStore(): Record<ProviderType, ProviderState> {
  return {
    minimax: makeDefaultProviderState('minimax'),
    glm: makeDefaultProviderState('glm'),
    deepseek: makeDefaultProviderState('deepseek'),
    openrouter: makeDefaultProviderState('openrouter'),
  };
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

  const [selectedProvider, setSelectedProvider] = useState<ProviderType>('openrouter');
  const [providerStore, setProviderStore] = useState<Record<ProviderType, ProviderState>>(defaultProviderStore);

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

    const nextStore = defaultProviderStore();
    let activeProvider: ProviderType | null = null;

    for (const item of cfgQuery.data.opencode_providers || []) {
      const type = providerTypeFromConfigName(item.name || '');
      if (!type) continue;
      const base = nextStore[type];
      const model = (item.models || [])[0] || base.model;
      const customModel = Boolean(model) && !base.availableModels.includes(model);

      nextStore[type] = {
        ...base,
        apiKey: item.api_key || '',
        apiKeySet: Boolean(item.api_key_set),
        clearApiKey: Boolean(item.clear_api_key),
        model,
        customModel,
      };

      if (item.enabled && activeProvider == null) {
        activeProvider = type;
      }
    }

    setProviderStore(nextStore);
    setSelectedProvider(activeProvider || 'openrouter');
  }, [cfgQuery.data]);

  const currentProviderState = providerStore[selectedProvider];

  const parsedProviders = useMemo(() => {
    const providers: OpencodeProvider[] = [];

    for (const type of PROVIDER_ORDER) {
      const state = providerStore[type];
      const meta = PROVIDER_META[type];
      providers.push({
        name: meta.providerId,
        enabled: type === selectedProvider,
        base_url: meta.baseUrl,
        api_key: state.apiKey.trim(),
        clear_api_key: state.clearApiKey,
        models: state.model.trim() ? [state.model.trim()] : [],
        headers: {},
        options: {},
      });
    }

    const selectedModel = providerStore[selectedProvider].model.trim();
    if (!selectedModel) {
      return { error: `${PROVIDER_META[selectedProvider].label} 需要选择模型。`, value: providers };
    }

    return { error: '', value: providers };
  }, [providerStore, selectedProvider]);

  const mergedConfig = useMemo<WebConfig | null>(() => {
    if (!cfgQuery.data) return null;
    const total = parseBudgetSeconds(totalBudget, 900, totalBudgetUnlimited);
    const selectedMeta = PROVIDER_META[selectedProvider];
    const selectedModel = providerStore[selectedProvider]?.model?.trim() || '';
    const selectedProviderApiKey = providerStore[selectedProvider]?.apiKey?.trim() || '';
    return {
      ...cfgQuery.data,
      openai_api_key: selectedProviderApiKey,
      openai_base_url: selectedMeta.baseUrl,
      openai_model: selectedModel,
      opencode_model: selectedModel,
      fuzz_time_budget: total,
      fuzz_use_docker: true,
      fuzz_docker_image: cfgQuery.data.fuzz_docker_image || 'auto',
      opencode_providers: parsedProviders.value,
    };
  }, [cfgQuery.data, totalBudget, totalBudgetUnlimited, selectedProvider, providerStore, parsedProviders.value]);

  const updateCurrentProvider = (patch: Partial<ProviderState>) => {
    setProviderStore((prev) => ({
      ...prev,
      [selectedProvider]: {
        ...prev[selectedProvider],
        ...patch,
      },
    }));
  };

  const handleFetchModels = async () => {
    const meta = PROVIDER_META[selectedProvider];
    updateCurrentProvider({ loadingModels: true, fetchError: '' });

    try {
      const out = await getOpencodeProviderModels(selectedProvider, {
        apiKey: currentProviderState.apiKey.trim(),
        baseUrl: meta.baseUrl,
      });
      const models = Array.from(new Set((out.models || []).map((x) => String(x || '').trim()).filter(Boolean)));
      if (!models.length) {
        updateCurrentProvider({ loadingModels: false, fetchError: '未获取到可用模型列表。' });
        return;
      }

      setProviderStore((prev) => {
        const current = prev[selectedProvider];
        const keepCurrent = current.customModel
          ? current.model
          : models.includes(current.model)
            ? current.model
            : models[0];
        return {
          ...prev,
          [selectedProvider]: {
            ...current,
            loadingModels: false,
            fetchError: '',
            availableModels: models,
            model: keepCurrent,
            customModel: current.customModel,
          },
        };
      });

      if ((out.source || '').toLowerCase() === 'remote') {
        setStatusType('success');
        setStatusText(`${meta.label} 模型列表已从远端更新。`);
      } else if (out.warning) {
        setStatusType('info');
        setStatusText(`${meta.label} 使用内置模型列表（${out.warning}）。`);
      } else {
        setStatusType('info');
        setStatusText(`${meta.label} 模型列表已更新（内置列表）。`);
      }
    } catch (e) {
      updateCurrentProvider({ loadingModels: false, fetchError: e instanceof Error ? e.message : '获取模型列表失败' });
    }
  };

  const handleSave = async () => {
    if (!mergedConfig) return;
    if (parsedProviders.error) {
      setStatusType('error');
      setStatusText(parsedProviders.error);
      return;
    }
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

          <Card variant="outlined" sx={{ p: 1.2 }}>
            <Stack spacing={1.5}>
              <Typography variant="subtitle1" fontWeight={600}>OpenCode Provider</Typography>

              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
                <FormControl fullWidth size="small">
                  <InputLabel id="provider-select-label">供应商</InputLabel>
                  <Select
                    labelId="provider-select-label"
                    label="供应商"
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(String(e.target.value) as ProviderType)}
                  >
                    {PROVIDER_ORDER.map((type) => (
                      <MenuItem key={type} value={type}>
                        {PROVIDER_META[type].label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Button
                  variant="outlined"
                  onClick={handleFetchModels}
                  disabled={currentProviderState.loadingModels}
                >
                  {currentProviderState.loadingModels ? '获取中...' : '获取模型列表'}
                </Button>
              </Stack>

              <TextField
                label="API Key"
                type="password"
                value={currentProviderState.apiKey}
                onChange={(e) => updateCurrentProvider({ apiKey: e.target.value, clearApiKey: false })}
                size="small"
                fullWidth
                helperText={currentProviderState.apiKeySet && !currentProviderState.apiKey ? '已保存密钥（留空则保持不变）' : undefined}
              />

              <FormControl fullWidth size="small">
                <InputLabel id="provider-model-label">模型</InputLabel>
                <Select
                  labelId="provider-model-label"
                  label="模型"
                  value={currentProviderState.customModel ? CUSTOM_MODEL_VALUE : currentProviderState.model}
                  onChange={(e) => {
                    const selected = String(e.target.value || '');
                    if (selected === CUSTOM_MODEL_VALUE) {
                      updateCurrentProvider({ customModel: true, model: '' });
                      return;
                    }
                    updateCurrentProvider({ customModel: false, model: selected });
                  }}
                >
                  {currentProviderState.availableModels.map((m) => (
                    <MenuItem key={m} value={m}>{m}</MenuItem>
                  ))}
                  <MenuItem value={CUSTOM_MODEL_VALUE}>自定义模型</MenuItem>
                </Select>
              </FormControl>

              {currentProviderState.customModel ? (
                <TextField
                  label="自定义模型"
                  value={currentProviderState.model}
                  onChange={(e) => updateCurrentProvider({ model: e.target.value })}
                  size="small"
                  fullWidth
                />
              ) : null}

              <FormControlLabel
                control={(
                  <Switch
                    checked={currentProviderState.clearApiKey}
                    onChange={(e) => updateCurrentProvider({ clearApiKey: e.target.checked })}
                  />
                )}
                label="保存时清空该 Provider 的 API Key"
              />

              {currentProviderState.fetchError ? <Alert severity="error">{currentProviderState.fetchError}</Alert> : null}
            </Stack>
          </Card>

          {parsedProviders.error ? <Alert severity="error">{parsedProviders.error}</Alert> : null}

          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              onClick={handleSave}
              disabled={saveCfg.isPending || !mergedConfig || Boolean(parsedProviders.error)}
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
