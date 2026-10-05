'use client';

import { useState } from 'react';
import { EmptyState } from '@/components/common/empty-state';
import { ErrorState } from '@/components/common/error-state';
import { LoadingState } from '@/components/common/loading-state';
import { PageHeader, SectionHeading } from '@/components/common/page-header';
import { useModelReport, useMonitoring } from '@/hooks';
import { formatDate, formatMonth } from '@/lib/format';
import { AlertsList } from './alerts-list';
import { BusinessTable } from './business-table';
import { DriftTable } from './drift-table';
import { MonthSelector } from './month-selector';
import { PatternSection } from './pattern-section';
import { PerformanceSection } from './performance-section';
import { PredictionDriftSection } from './prediction-drift';
import { RemediationTable } from './remediation-table';

export function MonitoringDashboard() {
  const monitoring = useMonitoring();
  // Only supplies the June recall-by-pattern baseline; the page works without it.
  const model = useModelReport();
  const [picked, setPicked] = useState<string | null>(null);

  const header = (
    <PageHeader
      title="Drift monitoring"
      description="Three live months after release: has the input data moved, have the scores moved, and is the model still catching fraud without declining genuine customers?"
    />
  );

  if (monitoring.isPending) {
    return (
      <>
        {header}
        <LoadingState label="Loading monitoring report" blocks={3} />
      </>
    );
  }
  if (monitoring.isError) {
    return (
      <>
        {header}
        <ErrorState error={monitoring.error} onRetry={() => monitoring.refetch()} title="Could not load the monitoring report" />
      </>
    );
  }

  const report = monitoring.data;
  if (report.months.length === 0) {
    return (
      <>
        {header}
        <EmptyState title="No monitored months yet" description="Run the monitoring pipeline (`make monitor`) to produce the report." />
      </>
    );
  }

  // Default to the most recent month.
  const month = report.months.find((m) => m.month === picked) ?? report.months[report.months.length - 1];
  const monthLabel = formatMonth(month.month);

  return (
    <>
      {header}
      <MonthSelector months={report.months} selected={month.month} onSelect={setPicked} />
      <p className="text-xs text-subtle">
        Model v{report.model_version}. Scores and metrics are compared with the June hold-out ({formatDate(report.reference.scores_and_metrics[0])} to{' '}
        {formatDate(report.reference.scores_and_metrics[1])}).
      </p>

      <AlertsList actions={month.actions} dataQuality={month.data_quality} monthLabel={monthLabel} />

      <SectionHeading title="Is the model still working?" description="Performance and business impact for all three months; the selected month is highlighted." />
      <PerformanceSection months={report.months} baseline={report.baseline} selected={month.month} />
      <div className="grid items-start gap-4 xl:grid-cols-2">
        <BusinessTable months={report.months} baseline={report.baseline.business} baselineWindow={report.reference.scores_and_metrics} selected={month.month} />
        <PatternSection current={month.recall_by_pattern} baseline={model.data?.champion.test_recall_by_pattern} monthLabel={monthLabel} />
      </div>

      <SectionHeading title={`What moved in ${monthLabel}?`} description="Input and score drift for the selected month." />
      <PredictionDriftSection drift={month.prediction_drift} histogram={report.score_histogram} month={month.month} monthLabel={monthLabel} />
      <DriftTable rows={month.data_drift} thresholds={report.thresholds} reference={report.reference.inputs} monthLabel={monthLabel} />

      <SectionHeading title="What to do about it" />
      <RemediationTable remediation={report.remediation} />
    </>
  );
}
