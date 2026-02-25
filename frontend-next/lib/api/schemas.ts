import { z } from 'zod';

export const configSchema = z.object({
  openai_api_key: z.string().optional().default(''),
  openai_api_key_set: z.boolean().optional(),
  openai_base_url: z.string().optional().default(''),
  openai_model: z.string().optional().default(''),
  opencode_model: z.string().optional().default(''),
  openrouter_api_key: z.string().optional().default(''),
  openrouter_base_url: z.string().optional().default(''),
  openrouter_model: z.string().optional().default(''),
  fuzz_time_budget: z.number().int().positive().default(900),
  fuzz_use_docker: z.boolean().default(true),
  fuzz_docker_image: z.string().default('auto'),
  oss_fuzz_dir: z.string().default(''),
  sherpa_git_mirrors: z.string().default(''),
  sherpa_docker_http_proxy: z.string().default(''),
  sherpa_docker_https_proxy: z.string().default(''),
  sherpa_docker_no_proxy: z.string().default(''),
  sherpa_docker_proxy_host: z.string().default('host.docker.internal'),
  version: z.number().int().default(1),
});

export const childStatusSchema = z.object({
  total: z.number().int().default(0),
  queued: z.number().int().default(0),
  running: z.number().int().default(0),
  success: z.number().int().default(0),
  error: z.number().int().default(0),
});

export const taskSummarySchema = z.object({
  job_id: z.string(),
  status: z.string(),
  repo: z.string().nullable().optional(),
  updated_at_iso: z.string().nullable().optional(),
  created_at_iso: z.string().nullable().optional(),
  children_status: childStatusSchema.default({ total: 0, queued: 0, running: 0, success: 0, error: 0 }),
  child_count: z.number().int().default(0),
  active_child_id: z.string().nullable().optional(),
  active_child_status: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  result: z.string().nullable().optional(),
});

export const taskListSchema = z.object({
  items: z.array(taskSummarySchema),
});

export const childJobSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  repo: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  result: z.any().optional(),
  log: z.string().optional().default(''),
  updated_at: z.number().optional(),
  started_at: z.number().nullable().optional(),
  finished_at: z.number().nullable().optional(),
});

export const taskDetailSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  repo: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  result: z.any().optional(),
  children_status: childStatusSchema.optional(),
  children: z.array(childJobSchema).optional().default([]),
});

export const systemSchema = z.object({
  ok: z.boolean().default(false),
  server_time_iso: z.string().optional(),
  uptime_sec: z.number().optional(),
  jobs: z
    .object({
      total: z.number().int().default(0),
      queued: z.number().int().default(0),
      running: z.number().int().default(0),
      success: z.number().int().default(0),
      error: z.number().int().default(0),
    })
    .default({ total: 0, queued: 0, running: 0, success: 0, error: 0 }),
  active_jobs: z.array(z.any()).optional().default([]),
  workers: z.object({ max: z.number().int().default(0) }).optional(),
});

export type WebConfig = z.infer<typeof configSchema>;
export type TaskSummary = z.infer<typeof taskSummarySchema>;
export type TaskDetail = z.infer<typeof taskDetailSchema>;
export type SystemStatus = z.infer<typeof systemSchema>;
