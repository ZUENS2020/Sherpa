import { z } from 'zod';

const normalizedErrorSchema = z
  .unknown()
  .transform((v) => {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (typeof v === 'object') {
      const obj = v as Record<string, unknown>;
      const detail = obj?.detail;
      if (typeof detail === 'string' && detail.trim()) return detail.trim();
      const message = obj?.message;
      if (typeof message === 'string' && message.trim()) return message.trim();
      if (Object.keys(obj).length === 0) return '';
      try {
        return JSON.stringify(obj);
      } catch {
        return String(v);
      }
    }
    return String(v);
  })
  .default('');

export const opencodeProviderSchema = z.object({
  name: z.string().default(''),
  enabled: z.boolean().default(true),
  base_url: z.string().default(''),
  api_key: z.string().optional().default(''),
  api_key_set: z.boolean().optional(),
  clear_api_key: z.boolean().optional().default(false),
  models: z.array(z.string()).default([]),
  headers: z.record(z.string(), z.string()).default({}),
  options: z.record(z.string(), z.any()).default({}),
});

export const configSchema = z.object({
  openai_api_key: z.string().optional().default(''),
  openai_api_key_set: z.boolean().optional(),
  openai_base_url: z.string().optional().default(''),
  openai_model: z.string().optional().default(''),
  opencode_model: z.string().optional().default(''),
  opencode_providers: z.array(opencodeProviderSchema).default([]),
  openrouter_api_key: z.string().optional().default(''),
  openrouter_base_url: z.string().optional().default(''),
  openrouter_model: z.string().optional().default(''),
  fuzz_time_budget: z.number().int().nonnegative().default(900),
  sherpa_run_unlimited_round_budget_sec: z.number().int().nonnegative().default(7200),
  sherpa_run_plateau_idle_growth_sec: z.number().int().min(30).max(86400).default(600),
  fuzz_use_docker: z.boolean().default(true),
  fuzz_docker_image: z.string().default('auto'),
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

export const vulnCandidateSchema = z
  .object({
    candidate_id: z.string().optional().default(''),
    validation_status: z.string().optional().default(''),
    classification: z.string().optional().default(''),
    confidence: z.number().optional().default(0),
    target_api: z.string().optional().default(''),
    target_name: z.string().optional().default(''),
    fuzzer: z.string().optional().default(''),
    sanitizer: z.string().optional().default(''),
    crash_type: z.string().optional().default(''),
    triage_label: z.string().optional().default(''),
    analysis_verdict: z.string().optional().default(''),
    reproduction_status: z.string().optional().default(''),
    reason: z.string().optional().default(''),
  })
  .passthrough()
  .default({});

export const frontierInputSchema = z.object({
  input_relpath: z.string().optional().default(''),
  size_bytes: z.number().int().optional().default(0),
  covered_function_count: z.number().int().optional().default(0),
  covered_region_count: z.number().int().optional().default(0),
  exec_time_us: z.number().int().optional().default(0),
  unique_frontier_functions: z.number().int().optional().default(0),
  nearby_uncovered_regions: z.number().int().optional().default(0),
  frontier_score: z.number().optional().default(0),
  covered_functions_sample: z.array(z.string()).optional().default([]),
  frontier_functions: z.array(z.object({
    name: z.string().optional().default(''),
    file: z.string().optional().default(''),
    line: z.number().int().optional().default(0),
    covered_region_count: z.number().int().optional().default(0),
    total_region_count: z.number().int().optional().default(0),
    uncovered_regions_nearby: z.number().int().optional().default(0),
    region_coverage_ratio: z.number().optional().default(0),
  })).optional().default([]),
  rationale: z.string().optional().default(''),
  repo_file_count: z.number().int().optional().default(0),
});

export const frontierFunctionSchema = z.object({
  name: z.string().optional().default(''),
  input_count: z.number().int().optional().default(0),
  input_relpaths: z.array(z.string()).optional().default([]),
});

export const frontierSummarySchema = z.object({
  version: z.number().int().optional().default(0),
  binary_hash: z.string().optional().default(''),
  replay_binary: z.string().optional().default(''),
  generated_at: z.number().int().optional().default(0),
  top_inputs: z.array(frontierInputSchema).optional().default([]),
  top_input_count: z.number().int().optional().default(0),
  top_frontier_functions: z.array(frontierFunctionSchema).optional().default([]),
  top_frontier_function_count: z.number().int().optional().default(0),
  failed_input_count: z.number().int().optional().default(0),
  pending_input_count: z.number().int().optional().default(0),
  covered_function_union_sample: z.array(z.string()).optional().default([]),
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
  error: normalizedErrorSchema.optional(),
  result: z.string().nullable().optional(),
  vuln_hunting_enabled: z.boolean().optional().default(false),
  security_priority_mode: z.boolean().optional().default(false),
  vuln_candidate_count: z.number().int().optional().default(0),
  crash_vuln_candidate_count: z.number().int().optional().default(0),
  latest_crash_vuln_candidate: vulnCandidateSchema.optional().default({}),
  fuzz_coverage_per_input_manifest_path: z.string().optional().default(''),
  fuzz_coverage_frontier_path: z.string().optional().default(''),
  fuzz_coverage_frontier_summary: frontierSummarySchema.optional().default({}),
  fuzz_coverage_replay_runtime_sec: z.number().optional().default(0),
  fuzz_coverage_replay_binary_hash: z.string().optional().default(''),
  fuzz_coverage_replay_binary_dir: z.string().optional().default(''),
  fuzz_coverage_replay_binary_count: z.number().int().optional().default(0),
  fuzz_coverage_replay_stage_success: z.boolean().optional().default(false),
  fuzz_coverage_replay_error: z.string().optional().default(''),
  fuzz_coverage_replay_manifest_fresh_for_current_binary: z.boolean().optional().default(false),
  fuzz_coverage_replay_queue_drained: z.boolean().optional().default(false),
  fuzz_coverage_replay_pending_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_failed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_processed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_total_inputs: z.number().int().optional().default(0),
});

export const taskListSchema = z.object({
  items: z.array(taskSummarySchema),
});

export const childJobSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  repo: z.string().nullable().optional(),
  error: normalizedErrorSchema.optional(),
  result: z.any().optional(),
  log: z.string().optional().default(''),
  vuln_hunting_enabled: z.boolean().optional().default(false),
  security_priority_mode: z.boolean().optional().default(false),
  vuln_candidate_count: z.number().int().optional().default(0),
  crash_vuln_candidate_count: z.number().int().optional().default(0),
  latest_crash_vuln_candidate: vulnCandidateSchema.optional().default({}),
  vuln_candidates_path: z.string().optional().default(''),
  crash_vuln_report_path: z.string().optional().default(''),
  fuzz_coverage_per_input_manifest_path: z.string().optional().default(''),
  fuzz_coverage_frontier_path: z.string().optional().default(''),
  fuzz_coverage_frontier_summary: frontierSummarySchema.optional().default({}),
  fuzz_coverage_replay_runtime_sec: z.number().optional().default(0),
  fuzz_coverage_replay_binary_hash: z.string().optional().default(''),
  fuzz_coverage_replay_binary_dir: z.string().optional().default(''),
  fuzz_coverage_replay_binary_count: z.number().int().optional().default(0),
  fuzz_coverage_replay_stage_success: z.boolean().optional().default(false),
  fuzz_coverage_replay_error: z.string().optional().default(''),
  fuzz_coverage_replay_manifest_fresh_for_current_binary: z.boolean().optional().default(false),
  fuzz_coverage_replay_queue_drained: z.boolean().optional().default(false),
  fuzz_coverage_replay_pending_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_failed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_processed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_total_inputs: z.number().int().optional().default(0),
  updated_at: z.number().optional(),
  started_at: z.number().nullable().optional(),
  finished_at: z.number().nullable().optional(),
});

export const taskDetailSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  repo: z.string().nullable().optional(),
  error: normalizedErrorSchema.optional(),
  result: z.any().optional(),
  children_status: childStatusSchema.optional(),
  children: z.array(childJobSchema).optional().default([]),
  vuln_hunting_enabled: z.boolean().optional().default(false),
  security_priority_mode: z.boolean().optional().default(false),
  vuln_candidate_count: z.number().int().optional().default(0),
  crash_vuln_candidate_count: z.number().int().optional().default(0),
  latest_crash_vuln_candidate: vulnCandidateSchema.optional().default({}),
  vuln_candidates_path: z.string().optional().default(''),
  crash_vuln_report_path: z.string().optional().default(''),
  fuzz_coverage_per_input_manifest_path: z.string().optional().default(''),
  fuzz_coverage_frontier_path: z.string().optional().default(''),
  fuzz_coverage_frontier_summary: frontierSummarySchema.optional().default({}),
  fuzz_coverage_replay_runtime_sec: z.number().optional().default(0),
  fuzz_coverage_replay_binary_hash: z.string().optional().default(''),
  fuzz_coverage_replay_binary_dir: z.string().optional().default(''),
  fuzz_coverage_replay_binary_count: z.number().int().optional().default(0),
  fuzz_coverage_replay_stage_success: z.boolean().optional().default(false),
  fuzz_coverage_replay_error: z.string().optional().default(''),
  fuzz_coverage_replay_manifest_fresh_for_current_binary: z.boolean().optional().default(false),
  fuzz_coverage_replay_queue_drained: z.boolean().optional().default(false),
  fuzz_coverage_replay_pending_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_failed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_processed_inputs: z.number().int().optional().default(0),
  fuzz_coverage_replay_total_inputs: z.number().int().optional().default(0),
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
  security: z
    .object({
      vuln_hunting_enabled: z.boolean().default(false),
      security_priority_mode: z.boolean().default(false),
      analysis_vuln_candidate_count: z.number().int().default(0),
      crash_vuln_candidate_count: z.number().int().default(0),
      latest_crash_vuln_candidate: vulnCandidateSchema.optional().default({}),
    })
    .optional()
    .default({
      vuln_hunting_enabled: false,
      security_priority_mode: false,
      analysis_vuln_candidate_count: 0,
      crash_vuln_candidate_count: 0,
      latest_crash_vuln_candidate: {},
    }),
  workers: z.object({ max: z.number().int().default(0) }).optional(),
});

export type WebConfig = z.infer<typeof configSchema>;
export type OpencodeProvider = z.infer<typeof opencodeProviderSchema>;
export type TaskSummary = z.infer<typeof taskSummarySchema>;
export type TaskDetail = z.infer<typeof taskDetailSchema>;
export type SystemStatus = z.infer<typeof systemSchema>;
