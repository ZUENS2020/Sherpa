import { afterEach, describe, expect, it, vi } from 'vitest';
import { putConfig } from './client';

describe('api client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('surfaces plain-text backend errors without JSON parse failures', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        headers: {
          get: (name: string) => (name.toLowerCase() === 'content-type' ? 'text/plain; charset=utf-8' : null),
        },
        text: async () => 'Internal Server Error',
      }),
    );

    await expect(putConfig({} as never)).rejects.toThrow('Internal Server Error');
  });

  it('keeps JSON success responses unchanged', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: {
          get: (name: string) => (name.toLowerCase() === 'content-type' ? 'application/json' : null),
        },
        text: async () => '{"ok":true}',
      }),
    );

    await expect(putConfig({} as never)).resolves.toEqual({ ok: true });
  });
});
