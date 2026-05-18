import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Sequence,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const palette = {
  bg: '#071012',
  panel: '#102327',
  panel2: '#0C1A1D',
  fg: '#E7F8EF',
  muted: '#8FB3A2',
  accent: '#66F2A5',
  warning: '#FFB86B',
  blue: '#3F7CFF',
  red: '#FF5C7A',
};

const fontDisplay = '"Avenir Next Condensed", "DIN Condensed", "Arial Narrow", sans-serif';
const fontBody = '"IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif';
const fontMono = '"JetBrains Mono", "SFMono-Regular", Menlo, monospace';

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const easeOut = Easing.bezier(0.16, 1, 0.3, 1);
const easeInOut = Easing.bezier(0.45, 0, 0.55, 1);

const fit = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, end], [0, 1], {...clamp, easing: easeOut});

const fadeWindow = (frame: number, start: number, end: number, fade = 12) => {
  const inV = interpolate(frame, [start, start + fade], [0, 1], clamp);
  const outV = interpolate(frame, [end - fade, end], [1, 0], clamp);
  return Math.min(inV, outV);
};

const Background: React.FC<{accent?: string}> = ({accent = palette.accent}) => {
  const frame = useCurrentFrame();
  const drift = interpolate(frame % 240, [0, 240], [0, 1], clamp);
  const pulse = interpolate(Math.sin(frame / 28), [-1, 1], [0.75, 1.08]);

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at ${18 + drift * 12}% 18%, ${accent}22 0, transparent 30%),
          radial-gradient(circle at ${82 - drift * 10}% 80%, ${palette.blue}1F 0, transparent 34%),
          linear-gradient(135deg, #071012 0%, #0B1618 52%, #061013 100%)`,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(231,248,239,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(231,248,239,0.035) 1px, transparent 1px)',
          backgroundSize: '72px 72px',
          transform: `translate(${drift * -72}px, ${drift * 72}px)`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          width: 980,
          height: 980,
          borderRadius: 999,
          right: -250,
          top: -260,
          border: `1px solid ${accent}30`,
          transform: `scale(${pulse})`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 80,
          bottom: 60,
          fontFamily: fontMono,
          fontSize: 120,
          letterSpacing: -6,
          color: '#FFFFFF08',
          transform: `translateX(${drift * 80}px)`,
        }}
      >
        SIGNAL / TRACE / VERIFY
      </div>
    </AbsoluteFill>
  );
};

const StageLabel: React.FC<{children: React.ReactNode; color?: string}> = ({
  children,
  color = palette.accent,
}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 12,
      padding: '10px 16px',
      border: `1px solid ${color}66`,
      background: `${color}16`,
      color,
      borderRadius: 999,
      fontFamily: fontMono,
      fontSize: 22,
      letterSpacing: 1.4,
      textTransform: 'uppercase',
      width: 'fit-content',
    }}
  >
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: 99,
        background: color,
        boxShadow: `0 0 24px ${color}`,
      }}
    />
    {children}
  </div>
);

const TitleBlock: React.FC<{
  label: string;
  title: string;
  body: string;
  frame: number;
  color?: string;
}> = ({label, title, body, frame, color = palette.accent}) => {
  const enter = fit(frame, 0, 28);
  return (
    <div
      style={{
        transform: `translateY(${interpolate(enter, [0, 1], [42, 0])}px)`,
        opacity: enter,
      }}
    >
      <StageLabel color={color}>{label}</StageLabel>
      <h1
        style={{
          margin: '30px 0 22px',
          maxWidth: 980,
          fontFamily: fontDisplay,
          fontSize: 116,
          lineHeight: 0.86,
          letterSpacing: -3,
          color: palette.fg,
          fontWeight: 900,
          textTransform: 'uppercase',
        }}
      >
        {title}
      </h1>
      <p
        style={{
          maxWidth: 760,
          margin: 0,
          fontFamily: fontBody,
          fontSize: 30,
          lineHeight: 1.35,
          color: palette.muted,
          fontWeight: 400,
        }}
      >
        {body}
      </p>
    </div>
  );
};

const NoiseToSignal: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = fadeWindow(frame, 0, 180);
  const signal = fit(frame, 20, 90);
  const lines = [
    'clone repo',
    'map APIs',
    'score risk',
    'build harness',
    'run fuzzer',
    'triage crash',
    'replay input',
    'archive evidence',
  ];

  return (
    <AbsoluteFill style={{opacity}}>
      <Background />
      <div style={{display: 'flex', height: '100%', padding: '110px 130px', boxSizing: 'border-box'}}>
        <div style={{flex: 1.05, display: 'flex', alignItems: 'center'}}>
          <TitleBlock
            frame={frame}
            label="Sherpa"
            title="Security research, staged."
            body="A vulnerability-oriented fuzzing control plane for public repositories."
          />
        </div>
        <div
          style={{
            flex: 0.95,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          <div
            style={{
              width: 620,
              height: 620,
              borderRadius: 44,
              border: `1px solid ${palette.accent}40`,
              background: 'linear-gradient(160deg, rgba(16,35,39,0.88), rgba(7,16,18,0.72))',
              boxShadow: `0 0 90px ${palette.accent}18`,
              padding: 34,
              boxSizing: 'border-box',
              transform: `rotate(${interpolate(signal, [0, 1], [-4, 0])}deg) scale(${interpolate(signal, [0, 1], [0.92, 1])})`,
              opacity: signal,
            }}
          >
            <div style={{fontFamily: fontMono, color: palette.accent, fontSize: 24, marginBottom: 24}}>
              /shared/output/task.trace
            </div>
            {lines.map((line, index) => {
              const row = fit(frame, 36 + index * 7, 66 + index * 7);
              return (
                <div
                  key={line}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    height: 52,
                    padding: '0 14px',
                    marginBottom: 10,
                    borderRadius: 14,
                    background: index % 2 === 0 ? '#FFFFFF08' : '#FFFFFF04',
                    opacity: row,
                    transform: `translateX(${interpolate(row, [0, 1], [46, 0])}px)`,
                  }}
                >
                  <span style={{fontFamily: fontBody, color: palette.fg, fontSize: 25}}>{line}</span>
                  <span style={{fontFamily: fontMono, color: index > 2 ? palette.warning : palette.accent}}>
                    {index > 2 ? 'queued' : 'ok'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const ProblemScene: React.FC = () => {
  const frame = useCurrentFrame();
  const local = frame - 180;
  const opacity = fadeWindow(frame, 180, 420);
  const cards = [
    ['Target drift', 'API names, harness names, and execution plans diverge.'],
    ['Build chaos', 'Every repository has its own dependency and linking traps.'],
    ['Seed blindness', 'Inputs can run forever while coverage stays flat.'],
    ['Crash noise', 'Harness bugs look like vulnerabilities until proven otherwise.'],
  ];

  return (
    <AbsoluteFill style={{opacity}}>
      <Background accent={palette.warning} />
      <div style={{height: '100%', padding: '115px 135px', boxSizing: 'border-box'}}>
        <TitleBlock
          frame={local}
          label="The problem"
          color={palette.warning}
          title="Fuzzing breaks before bugs do."
          body="Sherpa treats failure modes as first-class evidence, not as terminal errors."
        />
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 24,
            marginTop: 56,
            maxWidth: 1260,
          }}
        >
          {cards.map(([title, body], index) => {
            const enter = fit(local, 28 + index * 9, 62 + index * 9);
            return (
              <div
                key={title}
                style={{
                  border: `1px solid ${palette.warning}32`,
                  background: 'rgba(16,35,39,0.78)',
                  borderRadius: 28,
                  padding: 30,
                  minHeight: 150,
                  opacity: enter,
                  transform: `translateY(${interpolate(enter, [0, 1], [36, 0])}px)`,
                  boxShadow: `0 24px 80px rgba(0,0,0,0.22)`,
                }}
              >
                <div style={{fontFamily: fontMono, color: palette.warning, fontSize: 18, marginBottom: 14}}>
                  0{index + 1}
                </div>
                <div style={{fontFamily: fontDisplay, fontWeight: 900, color: palette.fg, fontSize: 48, lineHeight: 0.95}}>
                  {title}
                </div>
                <div style={{fontFamily: fontBody, color: palette.muted, fontSize: 23, lineHeight: 1.32, marginTop: 14}}>
                  {body}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const PlaneScene: React.FC = () => {
  const frame = useCurrentFrame();
  const local = frame - 420;
  const opacity = fadeWindow(frame, 420, 690);
  const gate = fit(local, 38, 98);

  return (
    <AbsoluteFill style={{opacity}}>
      <Background accent={palette.blue} />
      <div style={{height: '100%', padding: '100px 135px', boxSizing: 'border-box'}}>
        <TitleBlock
          frame={local}
          label="Architecture"
          color={palette.blue}
          title="Open strategy. Strict control."
          body="Agent output enters an advisory layer. The system normalizes target identity, seed profile, execution plan, and routing."
        />
        <div style={{position: 'absolute', right: 125, top: 185, width: 770, height: 650}}>
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 80,
              width: 310,
              borderRadius: 34,
              padding: 34,
              background: `${palette.blue}13`,
              border: `1px solid ${palette.blue}55`,
              opacity: fit(local, 34, 70),
            }}
          >
            <div style={{fontFamily: fontMono, color: palette.blue, fontSize: 21}}>ADVISORY PLANE</div>
            {['risk hypotheses', 'seed strategy', 'attack hints', 'harness ideas'].map((item, i) => (
              <div key={item} style={{fontFamily: fontBody, color: palette.fg, fontSize: 28, marginTop: 24 + i * 2}}>
                {item}
              </div>
            ))}
          </div>
          <div
            style={{
              position: 'absolute',
              right: 0,
              top: 80,
              width: 330,
              borderRadius: 34,
              padding: 34,
              background: `${palette.accent}13`,
              border: `1px solid ${palette.accent}55`,
              opacity: fit(local, 72, 108),
            }}
          >
            <div style={{fontFamily: fontMono, color: palette.accent, fontSize: 21}}>CONTROL PLANE</div>
            {['target identity', 'stage routing', 'verification mode', 'workflow context'].map((item, i) => (
              <div key={item} style={{fontFamily: fontBody, color: palette.fg, fontSize: 28, marginTop: 24 + i * 2}}>
                {item}
              </div>
            ))}
          </div>
          <div
            style={{
              position: 'absolute',
              left: 325,
              top: 230,
              width: 116,
              height: 4,
              background: `linear-gradient(90deg, ${palette.blue}, ${palette.accent})`,
              opacity: gate,
              boxShadow: `0 0 28px ${palette.accent}`,
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: 344,
              top: 182,
              width: 78,
              height: 78,
              borderRadius: 22,
              background: palette.panel,
              border: `1px solid ${palette.fg}28`,
              opacity: gate,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: palette.accent,
              fontFamily: fontMono,
              fontSize: 22,
              transform: `scale(${interpolate(gate, [0, 1], [0.8, 1])})`,
            }}
          >
            normalize
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const WorkflowScene: React.FC = () => {
  const frame = useCurrentFrame();
  const local = frame - 690;
  const opacity = fadeWindow(frame, 690, 960);
  const stages = ['analysis', 'plan', 'synthesize', 'build', 'run', 'replay', 'coverage'];

  return (
    <AbsoluteFill style={{opacity}}>
      <Background />
      <div style={{height: '100%', padding: '100px 120px', boxSizing: 'border-box'}}>
        <TitleBlock
          frame={local}
          label="Execution"
          title="Every stage leaves evidence."
          body="Kubernetes jobs isolate work, persist artifacts, and make the next decision observable."
        />
        <div style={{position: 'absolute', left: 135, right: 135, bottom: 135, height: 360}}>
          <div
            style={{
              position: 'absolute',
              left: 40,
              right: 40,
              top: 170,
              height: 3,
              background: '#FFFFFF18',
            }}
          />
          {stages.map((stage, i) => {
            const x = i * 244;
            const enter = fit(local, 42 + i * 10, 78 + i * 10);
            const active = fit(local, 70 + i * 10, 102 + i * 10);
            return (
              <div
                key={stage}
                style={{
                  position: 'absolute',
                  left: x,
                  top: i % 2 === 0 ? 70 : 185,
                  width: 210,
                  height: 132,
                  borderRadius: 26,
                  padding: 22,
                  boxSizing: 'border-box',
                  border: `1px solid ${palette.accent}${i < 4 ? '44' : '66'}`,
                  background: i < 4 ? 'rgba(16,35,39,0.82)' : `${palette.accent}12`,
                  opacity: enter,
                  transform: `translateY(${interpolate(enter, [0, 1], [32, 0])}px)`,
                  boxShadow: active > 0.4 ? `0 0 44px ${palette.accent}22` : 'none',
                }}
              >
                <div style={{fontFamily: fontMono, color: palette.accent, fontSize: 17}}>stage/{i + 1}</div>
                <div style={{fontFamily: fontDisplay, fontWeight: 900, color: palette.fg, fontSize: 40, marginTop: 18}}>
                  {stage}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const EvidenceScene: React.FC = () => {
  const frame = useCurrentFrame();
  const local = frame - 960;
  const opacity = fadeWindow(frame, 960, 1200);
  const rows = [
    ['crash_triage', 'harness_bug', palette.warning],
    ['signature', 'asan+stack+keyframe', palette.blue],
    ['repair_mode', 'fix-harness', palette.accent],
    ['verification', 'no AI in run/repro', palette.fg],
  ] as const;

  return (
    <AbsoluteFill style={{opacity}}>
      <Background accent={palette.warning} />
      <div style={{height: '100%', padding: '100px 130px', boxSizing: 'border-box'}}>
        <TitleBlock
          frame={local}
          label="Feedback loop"
          color={palette.warning}
          title="Crashes become decisions."
          body="Sherpa separates harness defects from vulnerability candidates, then routes the workflow with structured context."
        />
        <div
          style={{
            position: 'absolute',
            right: 125,
            top: 155,
            width: 735,
            borderRadius: 38,
            border: `1px solid ${palette.warning}36`,
            background: 'rgba(7,16,18,0.78)',
            padding: 34,
            boxSizing: 'border-box',
          }}
        >
          <div style={{fontFamily: fontMono, fontSize: 24, color: palette.warning, marginBottom: 28}}>
            fuzz/context/workflow_context.json
          </div>
          {rows.map(([k, v, color], i) => {
            const enter = fit(local, 30 + i * 14, 62 + i * 14);
            return (
              <div
                key={k}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '230px 1fr',
                  gap: 22,
                  alignItems: 'center',
                  minHeight: 72,
                  borderTop: `1px solid ${palette.fg}12`,
                  opacity: enter,
                  transform: `translateX(${interpolate(enter, [0, 1], [36, 0])}px)`,
                }}
              >
                <span style={{fontFamily: fontMono, color: palette.muted, fontSize: 22}}>{k}</span>
                <span style={{fontFamily: fontBody, color, fontSize: 30, fontWeight: 700}}>{v}</span>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const ClosingScene: React.FC = () => {
  const frame = useCurrentFrame();
  const local = frame - 1200;
  const opacity = fadeWindow(frame, 1200, 1350, 8);
  const enter = fit(local, 0, 42);
  const ring = interpolate(local, [0, 150], [0, 1], {...clamp, easing: easeInOut});

  return (
    <AbsoluteFill style={{opacity}}>
      <Background />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          textAlign: 'center',
          padding: 120,
          boxSizing: 'border-box',
        }}
      >
        <div
          style={{
            position: 'absolute',
            width: 680,
            height: 680,
            borderRadius: 999,
            border: `1px solid ${palette.accent}36`,
            transform: `scale(${0.82 + ring * 0.28})`,
            opacity: 0.8 - ring * 0.4,
          }}
        />
        <div
          style={{
            fontFamily: fontMono,
            color: palette.accent,
            fontSize: 28,
            letterSpacing: 3,
            opacity: enter,
            marginBottom: 28,
          }}
        >
          SHERPA
        </div>
        <div
          style={{
            fontFamily: fontDisplay,
            color: palette.fg,
            fontSize: 132,
            lineHeight: 0.86,
            maxWidth: 1120,
            fontWeight: 900,
            textTransform: 'uppercase',
            letterSpacing: -4,
            opacity: enter,
            transform: `translateY(${interpolate(enter, [0, 1], [50, 0])}px)`,
          }}
        >
          Vulnerability discovery. Controlled.
        </div>
        <div
          style={{
            marginTop: 42,
            fontFamily: fontBody,
            color: palette.muted,
            fontSize: 31,
            opacity: fit(local, 36, 80),
          }}
        >
          AI-guided risk analysis · deterministic verification · Kubernetes-scale fuzzing
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Caption: React.FC<{text: string; start: number; end: number}> = ({text, start, end}) => {
  const frame = useCurrentFrame();
  const opacity = fadeWindow(frame, start, end, 10);
  return (
    <div
      style={{
        position: 'absolute',
        left: 130,
        bottom: 58,
        right: 130,
        opacity,
        fontFamily: fontBody,
        fontSize: 27,
        lineHeight: 1.28,
        color: palette.fg,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        pointerEvents: 'none',
      }}
    >
      <span>{text}</span>
      <span style={{fontFamily: fontMono, color: palette.accent, fontSize: 18}}>sherpa.dev/control-plane</span>
    </div>
  );
};

export const SherpaPromo: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <Sequence from={0} durationInFrames={180}>
        <NoiseToSignal />
      </Sequence>
      <Sequence from={180} durationInFrames={240}>
        <ProblemScene />
      </Sequence>
      <Sequence from={420} durationInFrames={270}>
        <PlaneScene />
      </Sequence>
      <Sequence from={690} durationInFrames={270}>
        <WorkflowScene />
      </Sequence>
      <Sequence from={960} durationInFrames={240}>
        <EvidenceScene />
      </Sequence>
      <Sequence from={1200} durationInFrames={150}>
        <ClosingScene />
      </Sequence>
      <Caption text="AI proposes. The control plane decides. Verification stays deterministic." start={28} end={210} />
      <Caption text="Target identity, seed semantics, crashes, and coverage all become auditable artifacts." start={420} end={780} />
      <Caption text="Sherpa keeps the loop moving: plan, synthesize, build, run, replay, analyze." start={790} end={1160} />
    </AbsoluteFill>
  );
};
