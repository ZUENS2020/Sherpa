import { create } from 'zustand';

const ACTIVE_TASK_KEY = 'sherpa_active_task_id';

function readActiveTask(): string {
  if (typeof window === 'undefined') return '';
  try {
    return localStorage.getItem(ACTIVE_TASK_KEY) || '';
  } catch {
    return '';
  }
}

function writeActiveTask(taskId: string) {
  if (typeof window === 'undefined') return;
  try {
    if (taskId) localStorage.setItem(ACTIVE_TASK_KEY, taskId);
    else localStorage.removeItem(ACTIVE_TASK_KEY);
  } catch {
    // noop
  }
}

interface UiState {
  activeTaskId: string;
  logFilter: 'all' | 'warn' | 'error';
  logKeyword: string;
  autoScrollEnabled: boolean;
  hydrated: boolean;
  hydrate: () => void;
  setActiveTaskId: (taskId: string) => void;
  setLogFilter: (filter: 'all' | 'warn' | 'error') => void;
  setLogKeyword: (keyword: string) => void;
  setAutoScrollEnabled: (enabled: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTaskId: '',
  logFilter: 'all',
  logKeyword: '',
  autoScrollEnabled: true,
  hydrated: false,
  hydrate: () => {
    set({ activeTaskId: readActiveTask(), hydrated: true });
  },
  setActiveTaskId: (taskId: string) => {
    writeActiveTask(taskId);
    set({ activeTaskId: taskId, autoScrollEnabled: true });
  },
  setLogFilter: (filter) => set({ logFilter: filter }),
  setLogKeyword: (keyword) => set({ logKeyword: keyword }),
  setAutoScrollEnabled: (enabled) => set({ autoScrollEnabled: enabled }),
}));
