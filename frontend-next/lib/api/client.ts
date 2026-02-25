import {
  configSchema,
  systemSchema,
  taskDetailSchema,
  taskListSchema,
  type WebConfig,
  type SystemStatus,
  type TaskDetail,
  type TaskSummary,
} from './schemas';

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || '/api').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    cache: 'no-store',
  });

  const text = await res.text();
  const data = text ? JSON.parse(text) : {};

  if (!res.ok) {
    const detail = data?.detail || `HTTP ${res.status}`;
    throw new Error(String(detail));
  }

  return data as T;
}

export async function getConfig(): Promise<WebConfig> {
  const data = await request<unknown>('/config');
  return configSchema.parse(data);
}

export async function putConfig(cfg: WebConfig): Promise<{ ok: boolean }> {
  return request('/config', { method: 'PUT', body: JSON.stringify(cfg) });
}

export async function getSystem(): Promise<SystemStatus> {
  const data = await request<unknown>('/system');
  return systemSchema.parse(data);
}

export async function getTasks(): Promise<TaskSummary[]> {
  const data = await request<unknown>('/tasks');
  return taskListSchema.parse(data).items;
}

export async function getTask(jobId: string): Promise<TaskDetail> {
  const data = await request<unknown>(`/task/${encodeURIComponent(jobId)}`);
  return taskDetailSchema.parse(data);
}

export interface SubmitTaskInput {
  repoUrl: string;
  model?: string;
  totalTimeBudget: number;
  runTimeBudget: number;
  maxTokens: number;
}

export async function submitTask(input: SubmitTaskInput): Promise<{ job_id: string; status: string }> {
  return request('/task', {
    method: 'POST',
    body: JSON.stringify({
      jobs: [
        {
          code_url: input.repoUrl,
          model: input.model || undefined,
          max_tokens: input.maxTokens,
          docker: true,
          docker_image: 'auto',
          time_budget: input.totalTimeBudget,
          total_time_budget: input.totalTimeBudget,
          run_time_budget: input.runTimeBudget,
        },
      ],
      auto_init: true,
      build_images: true,
      force_build: false,
      force_clone: false,
    }),
  });
}
