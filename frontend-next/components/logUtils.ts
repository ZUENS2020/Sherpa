export type LogLevel = 'info' | 'warn' | 'error';

export function detectLevel(line: string): LogLevel {
  const txt = line.trim().toLowerCase();
  if (txt.startsWith('[error]') || txt.startsWith('error:')) return 'error';
  if (txt.startsWith('[warn]') || txt.startsWith('warning:')) return 'warn';
  return 'info';
}

export function filterLogLines(
  raw: string,
  level: 'all' | 'warn' | 'error',
  keyword: string,
): string[] {
  const kw = keyword.trim().toLowerCase();
  return raw
    .split(/\r?\n/)
    .filter((line) => {
      if (!line.trim()) return false;
      const detected = detectLevel(line);
      if (level === 'warn' && detected === 'info') return false;
      if (level === 'error' && detected !== 'error') return false;
      if (kw && !line.toLowerCase().includes(kw)) return false;
      return true;
    });
}
