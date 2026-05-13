'use client';

import { Box, Stack, Typography } from '@mui/material';
import { AppShell } from '@/components/AppShell';
import { ConfigPanel } from '@/components/ConfigPanel';
import { SystemOverviewCard } from '@/components/SystemOverviewCard';
import { useSystemQuery } from '@/lib/api/hooks';

export default function RunTaskPage() {
  const system = useSystemQuery();

  return (
    <AppShell
      eyebrow="Run / Controlled launch"
      title="TianHeng 任务发起"
      description="填写仓库地址和执行预算后提交。提交后可在监控台查看进度。"
      rail={
        <>
          <Typography className="suzuka-kicker">CONTROLLED INPUT</Typography>
          <Typography className="suzuka-kicker">AUDITABLE LAUNCH</Typography>
        </>
      }
    >
      <Stack direction="row" spacing={1.5} alignItems="stretch" sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <Box sx={{ width: 560, flexShrink: 0, minHeight: 0, overflow: 'auto', pr: 0.25 }}>
          <ConfigPanel />
        </Box>
        <Box sx={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
          <Stack spacing={1.5} sx={{ height: '100%', minHeight: 0, overflow: 'hidden' }}>
            <SystemOverviewCard
              data={system.data}
              error={system.isError ? (system.error as Error).message : undefined}
            />
            <Box className="suzuka-panel" sx={{ p: 2, flex: 1, minHeight: 0, overflow: 'auto' }}>
              <Stack spacing={1.25}>
                <Typography variant="h6">使用说明</Typography>
                <Typography variant="body2" color="text.secondary">
                  提交任务后，系统将自动执行 fuzz 工作流（plan → synthesize → build → run）。
                  任务状态实时更新，可在监控台查看进度和结果。
                </Typography>
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', md: 'repeat(3, minmax(0, 1fr))' },
                    gap: 1.5,
                  }}
                >
                  {[
                    ['01', '提交任务', '填写仓库 URL 与预算，点击发起任务。'],
                    ['02', '自动执行', '系统自动完成规划、脚手架生成、构建、运行全流程。'],
                    ['03', '监控观察', '回到监控台查看实时进度、覆盖率变化和漏洞发现。'],
                  ].map(([index, title, body]) => (
                    <Box key={index} className="suzuka-inset" sx={{ p: 1.5 }}>
                      <Typography className="suzuka-kicker">{index}</Typography>
                      <Typography variant="subtitle2" sx={{ mt: 0.75 }}>{title}</Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                        {body}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Stack>
            </Box>
          </Stack>
        </Box>
      </Stack>
    </AppShell>
  );
}
