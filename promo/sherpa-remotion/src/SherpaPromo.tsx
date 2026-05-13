import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Sequence,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const tokens = {
  green: '#006941',
  moss: '#E8F2EC',
  paper: '#F7F4EA',
  ink: '#101615',
  muted: '#63726B',
  hairline: '#D9E4DC',
  night: '#07110D',
  night2: '#0B1B14',
  mint: '#CDEEDB',
  amber: '#EAA845',
  red: '#D94A4A',
  blue: '#2D6CDF',
};

const line = {
  light: '#D8DDD9',
  dark: '#FFFFFF24',
};

type Lang = 'en' | 'zh';

const fontDisplay = '"Space Grotesk", "PingFang SC", "Hiragino Sans GB", "Avenir Next Condensed", "Arial Narrow", sans-serif';
const fontBody = '"Inter", "PingFang SC", "Hiragino Sans GB", "Avenir Next", "Segoe UI", sans-serif';
const fontMono = '"JetBrains Mono", "SFMono-Regular", Menlo, monospace';

const copy = {
  en: {
    hook: {
      kicker: 'Sherpa / risk-first fuzzing',
      title: 'AI hunts vulnerability paths. Fuzzing verifies them.',
      body: 'The agent proposes risks, targets, and seed strategies. Sherpa normalizes control state, builds harnesses, and validates evidence deterministically.',
      metrics: [
        ['signals', '8', tokens.green],
        ['stages traced', '7', tokens.blue],
        ['verification', 'no-AI', tokens.amber],
      ] as Array<[string, string, string]>,
      panelA: [
        ['security_mode', 'risk_first_v1', tokens.green],
        ['vuln_candidate_count', '24'],
        ['evidence_count', '71'],
        ['top_candidate', 'png_read_IDAT_data', tokens.blue],
      ] as Array<[string, string, string?]>,
      panelB: [
        ['score source', 'vulnerability risk'],
        ['coverage', 'reference only'],
        ['api exception', 'evidence-gated'],
        ['reason', 'deep decode + size math', tokens.green],
      ] as Array<[string, string, string?]>,
    },
    hunt: {
      kicker: 'vulnerability guidance',
      title: 'Not just “more coverage”. Risk has first priority.',
      body: 'Analysis records concrete evidence: source line, signal, confidence, exploitability, and why the target is worth validating.',
      evidence: [
        ['evidence_id', 'SEC-018'],
        ['source', 'pngrutil.c:1621'],
        ['attack path', 'IHDR → IDAT → row decode'],
        ['validation', 'harness + seeds + replay'],
      ],
    },
    contract: {
      kicker: 'control-plane contract',
      title: 'Agent freedom without state pollution.',
      body: 'Strategy stays open. Control state is system-owned, normalized, and auditable across every stage boundary.',
      nodes: [
        ['AI advisory', 'seed strategy / attack hints / harness ideas', tokens.blue, 'suggested'],
        ['Normalizer', 'target identity / seed profile / empty semantics', tokens.green, 'normalized'],
        ['Control plane', 'routing / context / execution plan', tokens.amber, 'system-owned'],
        ['Verifier', 'build / run / replay with no AI', tokens.red, 'runtime-derived'],
      ] as Array<[string, string, string, string]>,
    },
    verification: {
      kicker: 'function/path feedback',
      title: 'Run feedback gets specific.',
      body: 'The fuzzer returns function-level and path-level evidence so the next improvement targets exactly what is missing.',
      hintTitle: 'next_improvement_hint',
      hintRows: [
        ['cold path', 'png_handle_iCCP'],
        ['seed family', 'compressed_payload_variants'],
        ['mutation focus', 'chunk length + CRC'],
        ['routing', 'seed_replan, not blind parallelism'],
      ] as Array<[string, string, string?]>,
    },
    pipeline: {
      kicker: 'continuous validation loop',
      title: 'One candidate validates while the hunter keeps looking.',
      body: 'Sherpa turns vulnerability hypotheses into queued validation work without blocking the discovery engine.',
      steps: ['hunt', 'rank', 'synthesize', 'build', 'run', 'replay', 'triage', 'learn'],
      stateLabel: 'candidate state',
      stateValue: 'validating',
      runnerLabel: 'runner-up target',
      runnerValue: 'ready',
    },
    close: {
      title: 'Vulnerability discovery with a control plane.',
      body: 'AI explores. The system verifies. Every decision leaves an artifact.',
      pillars: [
        ['Risk-first analysis', 'evidence-backed target selection'],
        ['Strict control plane', 'normalized state and routing'],
        ['Deterministic verification', 'build, run, replay without AI'],
        ['Actionable feedback', 'function/path level improvement hints'],
      ],
    },
    footerA: 'Risk-first AI guidance, strict execution truth, deterministic validation.',
    footerB: 'Coverage feedback now speaks in functions, paths, seed families, and next actions.',
  },
  zh: {
    hook: {
      kicker: 'Sherpa / 风险优先模糊测试',
      title: 'AI 寻找漏洞路径，Fuzzing 负责验证。',
      body: 'Agent 提出风险、目标和种子策略。Sherpa 归一化控制状态，生成 harness，并用确定性流程验证证据。',
      metrics: [
        ['风险信号', '8', tokens.green],
        ['追踪阶段', '7', tokens.blue],
        ['验证方式', '无 AI', tokens.amber],
      ] as Array<[string, string, string]>,
      panelA: [
        ['安全模式', 'risk_first_v1', tokens.green],
        ['漏洞候选数', '24'],
        ['证据数量', '71'],
        ['最高优先目标', 'png_read_IDAT_data', tokens.blue],
      ] as Array<[string, string, string?]>,
      panelB: [
        ['评分来源', '漏洞风险'],
        ['覆盖率', '仅作参考'],
        ['内部 API 例外', '证据门控'],
        ['选择原因', '深层解码 + 尺寸计算', tokens.green],
      ] as Array<[string, string, string?]>,
    },
    hunt: {
      kicker: '漏洞引导',
      title: '不只是“更多覆盖”。风险拥有最高优先级。',
      body: '分析阶段记录可机读证据：源码位置、风险信号、置信度、可利用性，以及为什么值得验证。',
      evidence: [
        ['证据编号', 'SEC-018'],
        ['源码位置', 'pngrutil.c:1621'],
        ['攻击路径', 'IHDR → IDAT → 行解码'],
        ['验证方式', 'harness + 种子 + replay'],
      ],
    },
    contract: {
      kicker: '控制面契约',
      title: '给 Agent 自由，但不让状态被污染。',
      body: '策略保持开放。控制状态由系统维护、归一化，并在每个阶段边界可审计。',
      nodes: [
        ['AI 建议层', '种子策略 / 攻击提示 / harness 方案', tokens.blue, '建议'],
        ['归一化层', '目标身份 / 种子画像 / 空值语义', tokens.green, '归一化'],
        ['控制面', '路由 / 上下文 / 执行计划', tokens.amber, '系统维护'],
        ['验证器', 'build / run / replay，无 AI 参与', tokens.red, '运行派生'],
      ] as Array<[string, string, string, string]>,
    },
    verification: {
      kicker: '函数 / 路径反馈',
      title: 'Run 的反馈必须具体到函数和路径。',
      body: 'Fuzzer 返回函数级和路径级证据，让下一轮改进明确知道该补哪里。',
      hintTitle: '下一轮改进提示',
      hintRows: [
        ['冷路径', 'png_handle_iCCP'],
        ['种子族', 'compressed_payload_variants'],
        ['变异重点', 'chunk 长度 + CRC'],
        ['路由', 'seed_replan，不盲目加并发'],
      ] as Array<[string, string, string?]>,
    },
    pipeline: {
      kicker: '持续验证循环',
      title: '一个候选在验证，漏洞猎手继续寻找下一个。',
      body: 'Sherpa 把漏洞假设转成验证任务，不让发现引擎被单个候选阻塞。',
      steps: ['挖掘', '排序', '生成', '构建', '运行', '回放', '分诊', '学习'],
      stateLabel: '候选状态',
      stateValue: '验证中',
      runnerLabel: '备选目标',
      runnerValue: '就绪',
    },
    close: {
      title: '带控制面的漏洞发现系统。',
      body: 'AI 负责探索。系统负责验证。每一次决策都留下工件。',
      pillars: [
        ['风险优先分析', '有证据支撑的目标选择'],
        ['严格控制面', '状态和路由统一归一化'],
        ['确定性验证', 'build、run、replay 不由 AI 改写'],
        ['可执行反馈', '函数 / 路径级改进提示'],
      ],
    },
    footerA: '风险优先的 AI 引导，严格的执行真相，确定性的验证流程。',
    footerB: '覆盖反馈细化到函数、路径、种子族和下一步动作。',
  },
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);
const appleEase = Easing.bezier(0.22, 1, 0.36, 1);

const p = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, end], [0, 1], {...clamp, easing: ease});

const apple = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, end], [0, 1], {...clamp, easing: appleEase});

const appear = (frame: number, start: number, distance = 28) => {
  const v = apple(frame, start, start + 24);
  return {
    opacity: v,
    filter: `blur(${interpolate(v, [0, 1], [5, 0])}px)`,
    transform: `translateY(${interpolate(v, [0, 1], [distance, 0])}px) scale(${interpolate(v, [0, 1], [0.985, 1])})`,
  };
};

const sceneStyle = (frame: number, duration = 180): React.CSSProperties => {
  const intro = apple(frame, 0, 28);
  const outro = interpolate(frame, [duration - 12, duration], [1, 0], clamp);
  return {
    opacity: Math.min(intro, outro),
    transform: `translateY(${interpolate(intro, [0, 1], [18, 0])}px) scale(${interpolate(frame, [0, duration], [1.018, 1], clamp)})`,
    transformOrigin: '50% 50%',
  };
};

const clipReveal = (frame: number, start: number, end: number) => {
  const v = apple(frame, start, end);
  return `inset(0 ${interpolate(v, [0, 1], [100, 0])}% 0 0)`;
};

const card = (dark = false): React.CSSProperties => ({
  border: `1px solid ${dark ? line.dark : line.light}`,
  borderLeft: `4px solid ${tokens.green}`,
  background: dark ? '#0D1914' : '#FFFFFF',
  borderRadius: 6,
  boxShadow: dark ? '0 1px 0 rgba(255,255,255,0.08)' : '0 1px 0 rgba(16,22,21,0.08)',
});

const SpeedLineBackground: React.FC<{dark?: boolean; accent?: string}> = ({
  dark = false,
  accent = tokens.green,
}) => {
  const frame = useCurrentFrame();
  const drift = interpolate(frame % 180, [0, 180], [0, 1], clamp);
  const bg = dark ? tokens.night : '#F6F6F6';

  return (
    <AbsoluteFill style={{background: bg, overflow: 'hidden'}}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `linear-gradient(90deg, ${dark ? '#FFFFFF12' : '#00694114'} 1px, transparent 1px)`,
          backgroundSize: '76px 100%',
          transform: `translateX(${-drift * 76}px) skewX(-12deg)`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `linear-gradient(${dark ? '#FFFFFF0A' : '#1016150A'} 1px, transparent 1px)`,
          backgroundSize: '100% 64px',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 92,
          right: 92,
          top: 52,
          height: 1,
          background: accent,
          width: '15%',
          clipPath: clipReveal(frame, 0, 34),
        }}
      />
      <div
        style={{
          position: 'absolute',
          right: 92,
          top: 76,
          width: 380,
          height: 180,
          borderTop: `1px solid ${accent}${dark ? '66' : '44'}`,
          borderRight: `1px solid ${accent}${dark ? '66' : '44'}`,
          opacity: apple(frame, 10, 42),
          transform: `translateX(${interpolate(apple(frame, 10, 42), [0, 1], [34, 0])}px) skewX(-18deg)`,
        }}
      />
    </AbsoluteFill>
  );
};

const Kicker: React.FC<{children: React.ReactNode; dark?: boolean}> = ({children, dark = false}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 11,
      padding: '8px 12px',
      borderRadius: 4,
      border: `1px solid ${dark ? '#FFFFFF28' : tokens.green + '33'}`,
      borderLeft: `4px solid ${tokens.green}`,
      background: dark ? '#0D1914' : '#FFFFFF',
      color: dark ? tokens.mint : tokens.green,
      fontFamily: fontMono,
      fontSize: 18,
      letterSpacing: 1.5,
      textTransform: 'uppercase',
    }}
  >
    <span style={{width: 18, height: 1, background: 'currentColor'}} />
    {children}
  </div>
);

const Header: React.FC<{
  kicker: string;
  title: string;
  body: string;
  frame: number;
  dark?: boolean;
}> = ({kicker, title, body, frame, dark = false}) => (
  <div style={{...appear(frame, 0)}}>
    <Kicker dark={dark}>{kicker}</Kicker>
    <div
      style={{
        marginTop: 24,
        maxWidth: 960,
        fontFamily: fontDisplay,
        color: dark ? '#F4FFF8' : tokens.ink,
        fontWeight: 800,
        fontSize: 86,
        lineHeight: 0.92,
        letterSpacing: -3.2,
      }}
    >
      {title}
    </div>
    <div
      style={{
        marginTop: 18,
        maxWidth: 760,
        fontFamily: fontBody,
        color: dark ? '#B7C9BF' : tokens.muted,
        fontSize: 26,
        lineHeight: 1.28,
      }}
    >
      {body}
    </div>
  </div>
);

const Metric: React.FC<{
  label: string;
  value: string;
  tone?: string;
  delay: number;
  dark?: boolean;
}> = ({label, value, tone = tokens.green, delay, dark = false}) => {
  const frame = useCurrentFrame();
  const enter = p(frame, delay, delay + 16);
  return (
    <div
      style={{
        ...card(dark),
        padding: 22,
        opacity: enter,
        transform: `translateY(${interpolate(enter, [0, 1], [24, 0])}px)`,
      }}
    >
      <div style={{fontFamily: fontMono, fontSize: 15, color: dark ? '#AFC2B7' : tokens.muted}}>
        {label}
      </div>
      <div style={{fontFamily: fontDisplay, fontSize: 48, color: tone, fontWeight: 800, marginTop: 8}}>
        {value}
      </div>
    </div>
  );
};

const Chip: React.FC<{children: React.ReactNode; color?: string; dark?: boolean}> = ({
  children,
  color = tokens.green,
  dark = false,
}) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '7px 11px',
      borderRadius: 4,
      border: `1px solid ${color}44`,
      borderLeft: `3px solid ${color}`,
      background: dark ? '#0D1914' : '#FFFFFF',
      color,
      fontFamily: fontMono,
      fontSize: 15,
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </span>
);

const CodePanel: React.FC<{title: string; rows: Array<[string, string, string?]>; delay: number; dark?: boolean}> = ({
  title,
  rows,
  delay,
  dark = false,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{...card(dark), padding: 24, ...appear(frame, delay, 22)}}>
      <div style={{fontFamily: fontMono, color: dark ? tokens.mint : tokens.green, fontSize: 18, marginBottom: 16}}>
        {title}
      </div>
      {rows.map(([k, v, tone], i) => {
        const row = p(frame, delay + 8 + i * 5, delay + 22 + i * 5);
        return (
          <div
            key={`${k}-${v}`}
            style={{
              display: 'grid',
              gridTemplateColumns: '210px 1fr',
              gap: 16,
              alignItems: 'center',
              padding: '10px 0',
              borderTop: `1px solid ${dark ? '#FFFFFF12' : '#10161512'}`,
              opacity: row,
              transform: `translateX(${interpolate(row, [0, 1], [22, 0])}px)`,
            }}
          >
            <span style={{fontFamily: fontMono, color: dark ? '#8FA89B' : tokens.muted, fontSize: 17}}>{k}</span>
            <span style={{fontFamily: fontBody, color: tone || (dark ? '#F4FFF8' : tokens.ink), fontSize: 23, fontWeight: 700}}>
              {v}
            </span>
          </div>
        );
      })}
    </div>
  );
};

const HookScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].hook;
  return (
    <AbsoluteFill>
      <SpeedLineBackground />
      <div style={{height: '100%', padding: '76px 92px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <div style={{display: 'grid', gridTemplateColumns: '1.08fr 0.92fr', gap: 42, height: '100%'}}>
          <div>
            <Header
              frame={frame}
              kicker={c.kicker}
              title={c.title}
              body={c.body}
            />
            <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 18, marginTop: 42}}>
              {c.metrics.map(([label, value, tone], i) => (
                <Metric key={label} label={label} value={value} delay={18 + i * 7} tone={tone} />
              ))}
            </div>
            <div style={{display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 32, ...appear(frame, 38)}}>
              {['mem_oob', 'integer_overflow', 'path_traversal', 'uaf', 'authz_bypass', 'format_string'].map((x) => (
                <Chip key={x}>{x}</Chip>
              ))}
            </div>
          </div>
          <div style={{display: 'grid', gridTemplateRows: '1fr 1fr', gap: 22}}>
            <CodePanel
              title="fuzz/analysis_context.json"
              delay={22}
              rows={c.panelA}
            />
            <CodePanel
              title="selected_targets.json"
              delay={46}
              rows={c.panelB}
            />
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const HuntScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].hunt;
  const rows = [
    ['integer_overflow_candidate', 'size arithmetic near allocation', '0.86'],
    ['mem_oob_candidate', 'row transform writes to caller buffer', '0.81'],
    ['reachability_confidence', 'public decode entry + corpus examples', '0.74'],
    ['exploitability', 'attacker controls dimensions + chunks', '0.69'],
  ];
  return (
    <AbsoluteFill>
      <SpeedLineBackground dark />
      <div style={{height: '100%', padding: '72px 92px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <div style={{display: 'grid', gridTemplateColumns: '0.95fr 1.05fr', gap: 42}}>
          <Header
            frame={frame}
            dark
            kicker={c.kicker}
            title={c.title}
            body={c.body}
          />
          <div style={{display: 'grid', gap: 16, paddingTop: 12}}>
            {rows.map(([signal, reason, score], i) => {
              const enter = p(frame, 18 + i * 11, 36 + i * 11);
              return (
                <div
                  key={signal}
                  style={{
                    ...card(true),
                    padding: 22,
                    display: 'grid',
                    gridTemplateColumns: '1fr 90px',
                    gap: 16,
                    alignItems: 'center',
                    opacity: enter,
                    transform: `translateX(${interpolate(enter, [0, 1], [40, 0])}px)`,
                  }}
                >
                  <div>
                    <div style={{fontFamily: fontMono, color: tokens.mint, fontSize: 18}}>{signal}</div>
                    <div style={{fontFamily: fontBody, color: '#DCECE4', fontSize: 24, marginTop: 8}}>{reason}</div>
                  </div>
                  <div style={{fontFamily: fontDisplay, fontSize: 48, fontWeight: 800, color: tokens.mint}}>
                    {score}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div style={{position: 'absolute', left: 92, right: 92, bottom: 70, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16}}>
          {c.evidence.map(([k, v], i) => (
            <CodePanel key={k} title={k} rows={[['value', v, i === 0 ? tokens.mint : undefined]]} delay={62 + i * 5} dark />
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const ContractScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].contract;
  return (
    <AbsoluteFill>
      <SpeedLineBackground />
      <div style={{height: '100%', padding: '72px 92px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <Header
          frame={frame}
          kicker={c.kicker}
          title={c.title}
          body={c.body}
        />
        <div style={{position: 'absolute', left: 92, right: 92, top: 455, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20}}>
          {c.nodes.map(([title, body, color, chip], i) => {
            const enter = p(frame, 24 + i * 12, 44 + i * 12);
            return (
              <div
                key={title}
                style={{
                  ...card(),
                  minHeight: 250,
                  padding: 24,
                  opacity: enter,
                  transform: `translateY(${interpolate(enter, [0, 1], [42, 0])}px)`,
                  borderTop: `7px solid ${color}`,
                }}
              >
                <div style={{fontFamily: fontDisplay, fontSize: 43, color: tokens.ink, fontWeight: 800, lineHeight: 0.95}}>
                  {title}
                </div>
                <div style={{fontFamily: fontBody, color: tokens.muted, fontSize: 22, lineHeight: 1.25, marginTop: 18}}>
                  {body}
                </div>
                <div style={{marginTop: 22}}>
                  <Chip color={color}>{chip}</Chip>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{position: 'absolute', right: 92, top: 86, width: 560}}>
          <CodePanel
            title="workflow_context.json"
            delay={40}
            rows={[
              ['coverage_target_name', 'harness_png_decode'],
              ['coverage_target_api', 'png_image_finish_read'],
              ['seed_profile', 'decoder-binary'],
              ['families_missing', 'derived only'],
            ]}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const VerificationScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].verification;
  const bars = [
    ['pngrutil.c', 0.86, tokens.green],
    ['pngread.c', 0.64, tokens.blue],
    ['pngset.c', 0.37, tokens.amber],
    ['pngerror.c', 0.22, tokens.red],
  ] as const;
  return (
    <AbsoluteFill>
      <SpeedLineBackground dark accent={tokens.mint} />
      <div style={{height: '100%', padding: '72px 92px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <Header
          frame={frame}
          dark
          kicker={c.kicker}
          title={c.title}
          body={c.body}
        />
        <div style={{position: 'absolute', left: 92, right: 92, bottom: 72, display: 'grid', gridTemplateColumns: '1.05fr 0.95fr', gap: 28}}>
          <div style={{...card(true), padding: 28}}>
            <div style={{fontFamily: fontMono, color: tokens.mint, fontSize: 20, marginBottom: 24}}>
              coverage_by_function
            </div>
            {bars.map(([name, val, color], i) => {
              const enter = p(frame, 28 + i * 8, 46 + i * 8);
              return (
                <div key={name} style={{marginBottom: 22, opacity: enter}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', fontFamily: fontBody, color: '#F4FFF8', fontSize: 23}}>
                    <span>{name}</span>
                    <span style={{fontFamily: fontMono, color}}>{Math.round(val * 100)}%</span>
                  </div>
                  <div
                    style={{
                      height: 12,
                      borderRadius: 0,
                      background: '#FFFFFF12',
                      marginTop: 10,
                      overflow: 'hidden',
                      border: '1px solid #FFFFFF16',
                    }}
                  >
                    <div
                      style={{
                        width: `${interpolate(enter, [0, 1], [0, val * 100])}%`,
                        height: '100%',
                        borderRadius: 0,
                        background: color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <CodePanel
            dark
            title={c.hintTitle}
            delay={42}
            rows={c.hintRows}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const PipelineScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].pipeline;
  const steps = c.steps;
  return (
    <AbsoluteFill>
      <SpeedLineBackground />
      <div style={{height: '100%', padding: '72px 92px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <Header
          frame={frame}
          kicker={c.kicker}
          title={c.title}
          body={c.body}
        />
        <div style={{position: 'absolute', left: 92, right: 92, bottom: 130, height: 250}}>
          <div style={{position: 'absolute', left: 50, right: 50, top: 122, height: 2, background: tokens.green + '33'}} />
          {steps.map((step, i) => {
            const enter = p(frame, 25 + i * 6, 42 + i * 6);
            const active = p(frame, 38 + i * 6, 50 + i * 6);
            return (
              <div
                key={step}
                style={{
                  position: 'absolute',
                  left: i * 214,
                  top: i % 2 === 0 ? 30 : 132,
                  width: 186,
                  height: 118,
                  ...card(),
                  padding: 18,
                  opacity: enter,
                  transform: `translateY(${interpolate(active, [0, 1], [18, 0])}px)`,
                  borderColor: active > 0.7 ? tokens.green : tokens.hairline,
                }}
              >
                <div style={{fontFamily: fontMono, color: tokens.green, fontSize: 15}}>0{i + 1}</div>
                <div
                  style={{
                    fontFamily: fontDisplay,
                    fontWeight: 800,
                    fontSize: step.length > 8 ? 28 : 34,
                    color: tokens.ink,
                    marginTop: 18,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    lineHeight: 1,
                  }}
                >
                  {step}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{position: 'absolute', right: 92, top: 92, width: 505, display: 'grid', gap: 14}}>
          <Metric label={c.stateLabel} value={c.stateValue} delay={34} />
          <Metric label={c.runnerLabel} value={c.runnerValue} delay={42} tone={tokens.blue} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const CloseScene: React.FC<{lang: Lang}> = ({lang}) => {
  const frame = useCurrentFrame();
  const c = copy[lang].close;
  const pillars = c.pillars;
  return (
    <AbsoluteFill>
      <SpeedLineBackground dark />
      <div style={{height: '100%', padding: '76px 96px', boxSizing: 'border-box', ...sceneStyle(frame)}}>
        <div style={{display: 'grid', gridTemplateColumns: '0.98fr 1.02fr', gap: 50, height: '100%', alignItems: 'center'}}>
          <div style={{...appear(frame, 0)}}>
            <Kicker dark>Sherpa</Kicker>
            <div style={{fontFamily: fontDisplay, color: '#F4FFF8', fontSize: 104, fontWeight: 800, lineHeight: 0.88, letterSpacing: -4, marginTop: 28}}>
              {c.title}
            </div>
            <div style={{fontFamily: fontBody, color: '#B7C9BF', fontSize: 29, lineHeight: 1.28, marginTop: 28}}>
              {c.body}
            </div>
          </div>
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18}}>
            {pillars.map(([title, body], i) => (
              <div key={title} style={{...card(true), padding: 26, minHeight: 190, ...appear(frame, 20 + i * 8)}}>
                <div style={{fontFamily: fontMono, color: tokens.mint, fontSize: 17}}>pillar/{i + 1}</div>
                <div style={{fontFamily: fontDisplay, color: '#F4FFF8', fontWeight: 800, fontSize: 38, lineHeight: 0.98, marginTop: 18}}>
                  {title}
                </div>
                <div style={{fontFamily: fontBody, color: '#AFC2B7', fontSize: 22, lineHeight: 1.25, marginTop: 14}}>
                  {body}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Footer: React.FC<{text: string; start: number; end: number; dark?: boolean}> = ({text, start, end, dark = false}) => {
  const frame = useCurrentFrame();
  const opacity = Math.min(p(frame, start, start + 10), interpolate(frame, [end - 10, end], [1, 0], clamp));
  return (
    <div
      style={{
        position: 'absolute',
        left: 92,
        right: 92,
        bottom: 34,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        opacity,
        fontFamily: fontBody,
        color: dark ? '#DCECE4' : tokens.ink,
        fontSize: 22,
        pointerEvents: 'none',
      }}
    >
      <span>{text}</span>
      <span style={{fontFamily: fontMono, color: dark ? tokens.mint : tokens.green, fontSize: 16}}>
        design: Suzuka · engine: Remotion / HyperFrames
      </span>
    </div>
  );
};

export const SherpaPromo: React.FC<{lang?: Lang}> = ({lang = 'en'}) => {
  const {fps} = useVideoConfig();
  const scene = fps * 6;
  const c = copy[lang];
  return (
    <AbsoluteFill style={{backgroundColor: tokens.paper}}>
      <Sequence from={0} durationInFrames={scene}>
        <HookScene lang={lang} />
      </Sequence>
      <Sequence from={scene} durationInFrames={scene}>
        <HuntScene lang={lang} />
      </Sequence>
      <Sequence from={scene * 2} durationInFrames={scene}>
        <ContractScene lang={lang} />
      </Sequence>
      <Sequence from={scene * 3} durationInFrames={scene}>
        <VerificationScene lang={lang} />
      </Sequence>
      <Sequence from={scene * 4} durationInFrames={scene}>
        <PipelineScene lang={lang} />
      </Sequence>
      <Sequence from={scene * 5} durationInFrames={scene}>
        <CloseScene lang={lang} />
      </Sequence>
      <Footer text={c.footerA} start={0} end={scene * 2} />
      <Footer text={c.footerB} start={scene * 3} end={scene * 5} dark />
    </AbsoluteFill>
  );
};

export const SherpaPromoZh: React.FC = () => <SherpaPromo lang="zh" />;
