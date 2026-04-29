import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TaskProgressPanel } from './TaskProgressPanel';
import type { TaskDetail } from '@/lib/api/schemas';

describe('TaskProgressPanel', () => {
  it('renders function gap and path frontier summaries from run feedback', () => {
    const detail = {
      job_id: 'job-1',
      status: 'running',
      children_status: { total: 1, queued: 0, running: 1, success: 0, error: 0 },
      children: [
        {
          job_id: 'child-1',
          status: 'running',
          result: {
            last_step: 'coverage-analysis',
            coverage_loop_round: 2,
            coverage_loop_max_rounds: 8,
            coverage_target_api: 'png_read_info',
          },
          fuzz_coverage_run_feedback_summary: {
            function_gap_count: 4,
            path_frontier_count: 2,
            top_function_gaps: [
              {
                name: 'png_read_info',
                kind: 'partial',
                region_coverage_ratio: 0.42,
                file: 'pngread.c',
                line: 123,
              },
            ],
            top_path_frontiers: [
              {
                input_relpath: 'corpus/seed-1',
                frontier_score: 0.91,
                covered_function_count: 7,
                covered_region_count: 19,
              },
            ],
          },
        },
      ],
    } as unknown as TaskDetail;

    render(<TaskProgressPanel detail={detail} />);

    expect(screen.getByText('Run Feedback')).toBeTruthy();
    expect(screen.getByText('4 gaps · 2 paths')).toBeTruthy();
    expect(screen.getByText('函数缺口')).toBeTruthy();
    expect(screen.getByText(/png_read_info · partial · 0\.42 · pngread\.c:123/)).toBeTruthy();
    expect(screen.getByText('路径前沿')).toBeTruthy();
    expect(screen.getByText(/corpus\/seed-1 · score 0\.91 · fn 7 · regions 19/)).toBeTruthy();
  });
});
