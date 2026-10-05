'use client';

import { StatusBadge } from '@/components/common/status-badge';
import { formatMonth, formatNumber, formatStat } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { MonitoringMonth } from '@/types';

interface MonthSelectorProps {
  months: MonitoringMonth[];
  selected: string;
  onSelect: (month: string) => void;
}

/** The three live months side by side; choosing one drives the detail sections below. */
export function MonthSelector({ months, selected, onSelect }: MonthSelectorProps) {
  return (
    <div role="group" aria-label="Month" className="grid gap-3 sm:grid-cols-3">
      {months.map((m) => {
        const active = m.month === selected;
        const alerts = m.actions.filter((a) => a.severity === 'alert').length;
        const warnings = m.actions.filter((a) => a.severity === 'warn').length;
        return (
          <button
            key={m.month}
            type="button"
            aria-pressed={active}
            onClick={() => onSelect(m.month)}
            className={cn(
              'rounded-card border bg-card p-4 text-left transition-colors',
              active ? 'border-primary ring-2 ring-primary/30' : 'hover:bg-muted/60'
            )}
          >
            <span className="flex items-center justify-between gap-2">
              <span className="text-base font-semibold">{formatMonth(m.month)}</span>
              <StatusBadge status={m.status} />
            </span>
            <span className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <span>
                <span className="block text-muted-foreground">PR-AUC</span>
                <span className="tabular block text-sm font-semibold">{formatStat(m.performance.pr_auc, 2)}</span>
              </span>
              <span>
                <span className="block text-muted-foreground">Transactions</span>
                <span className="tabular block text-sm font-semibold">{formatNumber(m.rows)}</span>
              </span>
              <span>
                <span className="block text-muted-foreground">Alerts / warnings</span>
                <span className="tabular block text-sm font-semibold">
                  {alerts} / {warnings}
                </span>
              </span>
            </span>
            <span className="mt-2 block text-xs text-subtle">{active ? 'Shown below' : 'Select to show below'}</span>
          </button>
        );
      })}
    </div>
  );
}
