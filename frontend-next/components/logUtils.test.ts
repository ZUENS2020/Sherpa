import { describe, expect, it } from 'vitest';
import { detectLevel, filterLogLines } from './logUtils';

describe('logUtils', () => {
  it('detects explicit levels only', () => {
    expect(detectLevel('[warn] something')).toBe('warn');
    expect(detectLevel('error: failed')).toBe('error');
    expect(detectLevel('plain info')).toBe('info');
  });

  it('filters by level and keyword', () => {
    const raw = ['[warn] one', 'error: two', 'hello world'].join('\n');
    expect(filterLogLines(raw, 'error', '')).toEqual(['error: two']);
    expect(filterLogLines(raw, 'warn', '')).toEqual(['[warn] one', 'error: two']);
    expect(filterLogLines(raw, 'all', 'hello')).toEqual(['hello world']);
  });
});
