'use client';

import { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import * as Sentry from '@sentry/nextjs';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#0f5ad8' },
    secondary: { main: '#00695f' },
    error: { main: '#c62828' },
    warning: { main: '#ed6c02' },
    success: { main: '#2e7d32' },
    background: { default: '#f5f7fb', paper: '#ffffff' },
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: '"DM Sans", "PingFang SC", "Microsoft YaHei", sans-serif',
  },
});

let sentryReady = false;
function initSentryIfNeeded() {
  const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
  if (!dsn || sentryReady) return;
  Sentry.init({
    dsn,
    tracesSampleRate: 0.1,
    enabled: true,
  });
  sentryReady = true;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() =>
    new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 1_000,
          retry: 1,
          refetchOnWindowFocus: false,
        },
      },
    }),
  );

  useEffect(() => {
    initSentryIfNeeded();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
}
