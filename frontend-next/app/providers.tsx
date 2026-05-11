'use client';

import { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import * as Sentry from '@sentry/nextjs';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#006941', contrastText: '#f5f0e6' },
    secondary: { main: '#191c1a', contrastText: '#f5f0e6' },
    error: { main: '#9f1d1d' },
    warning: { main: '#a85f00' },
    success: { main: '#006941' },
    background: { default: '#f5f0e6', paper: '#fffaf0' },
    text: { primary: '#191c1a', secondary: '#5f665f' },
    divider: '#191c1a',
  },
  shape: { borderRadius: 4 },
  typography: {
    fontFamily: '"Inter", "PingFang SC", "Microsoft YaHei", sans-serif',
    h4: {
      fontFamily: '"Space Grotesk", "Inter", "PingFang SC", sans-serif',
      fontWeight: 700,
      letterSpacing: '-0.04em',
    },
    h6: {
      fontFamily: '"Space Grotesk", "Inter", "PingFang SC", sans-serif',
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    subtitle2: {
      fontFamily: '"Space Grotesk", "Inter", "PingFang SC", sans-serif',
      fontWeight: 700,
      letterSpacing: '-0.015em',
    },
    caption: {
      letterSpacing: '0.02em',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          position: 'relative',
          border: '1px solid #191c1a',
          borderRadius: 4,
          boxShadow: '8px 8px 0 rgba(25, 28, 26, 0.10)',
          overflow: 'hidden',
          transition: 'transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease',
          '&::before': {
            content: '""',
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: 4,
            backgroundColor: '#006941',
          },
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: 16,
          '&:last-child': { paddingBottom: 16 },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          fontWeight: 800,
          boxShadow: 'none',
          transition: 'transform 150ms ease, box-shadow 150ms ease, background-color 150ms ease',
          '&:hover': {
            transform: 'translate(-1px, -1px)',
            boxShadow: '4px 4px 0 rgba(25, 28, 26, 0.18)',
          },
        },
        outlined: {
          borderColor: '#191c1a',
          color: '#191c1a',
          backgroundColor: '#fffaf0',
        },
        contained: {
          backgroundColor: '#006941',
          color: '#f5f0e6',
          '&:hover': {
            backgroundColor: '#005536',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 3,
          borderColor: '#191c1a',
          fontWeight: 800,
          letterSpacing: '0.045em',
          textTransform: 'uppercase',
          backgroundColor: '#fffaf0',
        },
        colorWarning: {
          backgroundColor: '#f1c44e',
          color: '#191c1a',
        },
        colorSuccess: {
          backgroundColor: '#006941',
          color: '#f5f0e6',
        },
        colorError: {
          backgroundColor: '#9f1d1d',
          color: '#fffaf0',
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          backgroundColor: '#fffaf0',
          '& fieldset': { borderColor: '#191c1a' },
          '&:hover fieldset': { borderColor: '#006941' },
          '&.Mui-focused fieldset': { borderColor: '#006941', borderWidth: 2 },
        },
      },
    },
    MuiInputLabel: {
      styleOverrides: {
        root: {
          color: '#5f665f',
          '&.Mui-focused': { color: '#006941' },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          border: '1px solid #191c1a',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          height: 8,
          borderRadius: 0,
          backgroundColor: '#e4ddce',
          border: '1px solid #191c1a',
        },
        bar: {
          backgroundColor: '#006941',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          height: 3,
          backgroundColor: '#006941',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 800,
          letterSpacing: '0.055em',
          textTransform: 'uppercase',
          color: '#5f665f',
          '&.Mui-selected': {
            color: '#006941',
          },
        },
      },
    },
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
