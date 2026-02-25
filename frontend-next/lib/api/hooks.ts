'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getConfig,
  getSystem,
  getTask,
  getTasks,
  putConfig,
  submitTask,
  type SubmitTaskInput,
} from './client';
import type { WebConfig } from './schemas';

export function useSystemQuery() {
  return useQuery({
    queryKey: ['system'],
    queryFn: getSystem,
    refetchInterval: 2000,
  });
}

export function useTasksQuery() {
  return useQuery({
    queryKey: ['tasks'],
    queryFn: getTasks,
    refetchInterval: 3000,
  });
}

export function useTaskDetailQuery(taskId: string | null) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: () => getTask(taskId as string),
    enabled: Boolean(taskId),
    refetchInterval: 2000,
  });
}

export function useConfigQuery() {
  return useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
  });
}

export function useSaveConfigMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cfg: WebConfig) => putConfig(cfg),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['config'] });
      void qc.invalidateQueries({ queryKey: ['system'] });
    },
  });
}

export function useSubmitTaskMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: SubmitTaskInput) => submitTask(input),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['tasks'] });
      void qc.invalidateQueries({ queryKey: ['system'] });
    },
  });
}
