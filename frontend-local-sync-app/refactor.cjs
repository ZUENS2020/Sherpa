const fs = require('fs');
let content = fs.readFileSync('src/App.tsx', 'utf8');

const replacements = [
  ['42<span className="ml-1 text-xs">m</span>', '{systemStatus?.overview?.avg_fuzz_time || \'42m\'}'],
  ['1,024\n                  </p>', '{systemStatus?.overview?.active_agents || \'1,024\'}\n                  </p>'],
  ['98.2<span className="text-lg">%</span>', '{systemStatus?.overview?.cluster_health || \'98.2\'}<span className="text-lg">%</span>'],
  ['+0.5% ▲', '{systemStatus?.overview?.cluster_health_trend || \'+0.5% ▲\'}'],
  ['42<span className="text-lg">/hr</span>', '{systemStatus?.overview?.crash_triage_rate || \'42\'}<span className="text-lg">/hr</span>'],
  ['+2% ▲', '{systemStatus?.overview?.crash_triage_rate_trend || \'+2% ▲\'}'],
  ['1.2<span className="text-lg">k</span>', '{systemStatus?.overview?.harnesses_synthesized || \'1.2\'}<span className="text-lg">k</span>'],
  ['+15% ▲', '{systemStatus?.overview?.harnesses_synthesized_trend || \'+15% ▲\'}'],
  ['68.4<span className="text-lg">%</span>', '{systemStatus?.overview?.avg_coverage || \'68.4\'}<span className="text-lg">%</span>'],
  ['+1.2% ▲', '{systemStatus?.overview?.avg_coverage_trend || \'+1.2% ▲\'}'],
  ['4.2M / hr', '{systemStatus?.telemetry?.llm_token_usage || \'4.2M / hr\'}'],
  ['>Stable<', '>{systemStatus?.telemetry?.llm_token_status || \'Stable\'}<'],
  ['88% CAP', '{systemStatus?.telemetry?.k8s_pod_capacity || \'88% CAP\'}'],
  ['Expansion Req', '{systemStatus?.telemetry?.k8s_pod_status || \'Expansion Req\'}'],
  ['99.9% SLI', '{systemStatus?.telemetry?.fastapi_gateway || \'99.9% SLI\'}'],
  ['Encrypted', '{systemStatus?.telemetry?.fastapi_status || \'Encrypted\'}'],
  ['data={performanceData}', 'data={systemStatus?.telemetry?.performance_series || performanceData}'],
  ['0.42%', '{systemStatus?.execution?.summary?.failure_rate || \'0.42%\'}'],
  ['1,204\n                          </span>', '{systemStatus?.execution?.summary?.fuzzing_jobs_24h || \'1,204\'}\n                          </span>'],
  ['68%\n                          </span>', '{systemStatus?.execution?.summary?.cluster_load_peak || \'68%\'}\n                          </span>'],
  ['142 <span className="text-xs">REPOS</span>', '{systemStatus?.execution?.summary?.repos_queued || \'142\'} <span className="text-xs">REPOS</span>'],
  ['482 <span className="text-xs">ms</span>', '{systemStatus?.execution?.summary?.avg_triage_time_ms || \'482\'} <span className="text-xs">ms</span>'],
  ['99.58 <span className="text-xs">%</span>', '{systemStatus?.execution?.summary?.success_ratio || \'99.58\'} <span className="text-xs">%</span>'],
  ['{systemStatus?.jobs?.total || \'1,204\'}', '{systemStatus?.tasks_tab_metrics?.total_jobs || \'1,204\'}'],
  ['{systemStatus?.performance?.throughput || \'84.2\'}', '{systemStatus?.tasks_tab_metrics?.execs_per_sec || \'84.2\'}'],
  ['{systemStatus?.jobs?.success_rate || \'98.4\'}', '{systemStatus?.tasks_tab_metrics?.success_rate || \'98.4\'}'],
  ['{systemStatus?.jobs?.failed || \'02\'}', '{systemStatus?.tasks_tab_metrics?.failed_tasks || \'02\'}']
];

replacements.forEach(([search, replace]) => {
  content = content.replace(search, replace);
});

content = content.replace(
  /const isWarning = i === 4 \|\| i === 18 \|\| i === 29;/g,
  'const matrix = systemStatus?.telemetry?.agent_health_matrix;\n                          const isWarning = matrix ? matrix[i] === 0 : (i === 4 || i === 18 || i === 29);'
);

fs.writeFileSync('src/App.tsx', content);
console.log('Refactored App.tsx');
