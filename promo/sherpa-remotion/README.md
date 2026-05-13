# Sherpa Promo Video

Remotion 宣传片工程，视觉与脚本按 HyperFrames 的信息密度方法组织，并套用 Suzuka Design System：

- `DESIGN.md`: 视觉身份、色彩、字体、运动原则
- `STORYBOARD.md`: 36 秒脚本与分镜
- `src/SherpaPromo.tsx`: Remotion composition
- Compositions:
  - `SherpaPromo`: English
  - `SherpaPromoZh`: Chinese

## 内容定位

主题：`Vulnerability discovery. Controlled.`

表达重点：

- AI 负责风险分析、目标建议、harness 策略。
- 控制面负责 target identity、seed profile、stage routing、workflow context 归一化。
- run/repro 验证阶段保持确定性，不让 AI 污染验证结论。
- Kubernetes 阶段作业保留所有可审计产物。

## 本地预览

```bash
npm install
npm run studio
```

## 渲染

```bash
npm run still
npm run render
npm run render:zh
```

如果首次渲染时 Remotion 下载 Headless Shell 失败，原因通常是到 `storage.googleapis.com` 的网络连接中断。解决方式：

```bash
npx remotion browser ensure
npm run still
```

或安装本机 Chrome/Chromium 后按 Remotion 文档配置浏览器可执行路径。

## 已验证

- `npm install` 成功。
- `npx tsc --noEmit` 成功。
- 使用本机 Chrome 渲染成功：
  - still: `out/frame-90.png`
  - English video: `out/sherpa-promo.mp4`
  - Chinese video: `out/sherpa-promo-zh.mp4`
