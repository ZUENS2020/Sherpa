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
      description="面向操作员提交新仓库和调整执行预算。页面只处理配置与启动动作；提交后可切回监控台观察阶段、日志和漏洞候选。"
      rail={(
        <>
          <Typography className="suzuka-kicker">CONTROLLED INPUT</Typography>
          <Typography className="suzuka-kicker">AUDITABLE LAUNCH</Typography>
        </>
      )}
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
                <Typography variant="h6">工业操作约束</Typography>
                <Typography variant="body2" color="text.secondary">
                  任务发起页只暴露仓库地址、总预算、单轮预算和平台期窗口。验证阶段仍由后端控制，run/repro 不引入 AI 判定，避免执行结果被策略层污染。
                </Typography>
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', md: 'repeat(3, minmax(0, 1fr))' },
                    gap: 1.5,
                  }}
                >
                  {[
                    ['01', '输入收口', '仓库 URL 与预算集中填写，避免监控页误提交。'],
                    ['02', '执行隔离', '配置保存与任务提交显式分离，可审计、可回滚。'],
                    ['03', '监控分流', '提交成功后记录 active task，监控台自动跟踪。'],
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
