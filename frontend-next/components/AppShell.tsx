'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Box, Button, Stack, Typography } from '@mui/material';

const navItems = [
  { href: '/', label: '监控台', code: 'MONITOR' },
  { href: '/run', label: '跑任务', code: 'RUN' },
];

interface AppShellProps {
  eyebrow: string;
  title: string;
  description: string;
  rail?: React.ReactNode;
  dense?: boolean;
  children: React.ReactNode;
}

export function AppShell({ eyebrow, title, description, rail, dense = false, children }: AppShellProps) {
  const pathname = usePathname();

  return (
    <Box className="suzuka-shell">
      <Stack className="suzuka-frame" spacing={dense ? 1.25 : 1.5}>
        <Box className="suzuka-topbar">
          <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="space-between">
            <Stack direction="row" spacing={1.25} alignItems="center">
              <Box className="suzuka-mark">TH</Box>
              <Stack spacing={0}>
                <Typography className="suzuka-kicker">XDU TianHeng Lab</Typography>
                <Typography variant="caption" color="text.secondary">Industrial vulnerability hunting console</Typography>
              </Stack>
            </Stack>
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
              {navItems.map((item) => {
                const selected = pathname === item.href;
                return (
                  <Button
                    key={item.href}
                    component={Link}
                    href={item.href}
                    variant={selected ? 'contained' : 'outlined'}
                    size="small"
                  >
                    {item.code} / {item.label}
                  </Button>
                );
              })}
            </Stack>
          </Stack>
        </Box>

        <Box className="suzuka-hero" sx={{ px: { xs: 2, md: 3 }, py: { xs: 2, md: 2.5 } }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="flex-end"
            spacing={2}
            sx={{ position: 'relative', zIndex: 1 }}
          >
            <Stack spacing={0.75}>
              <Typography className="suzuka-kicker">{eyebrow}</Typography>
              <Typography variant="h4">{title}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 780 }}>
                {description}
              </Typography>
            </Stack>
            {rail ? (
              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                {rail}
              </Stack>
            ) : null}
          </Stack>
        </Box>

        {children}
      </Stack>
    </Box>
  );
}
