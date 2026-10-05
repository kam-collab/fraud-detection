import { CheckCircle2 } from 'lucide-react';
import { StatusBadge } from '@/components/common/status-badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { DataQuality, MonitoringAction } from '@/types';

const ORDER = { alert: 0, warn: 1, ok: 2 } as const;

/** Monitoring alerts for the month, most severe first, each with its recommended action. */
export function AlertsList({ actions, dataQuality, monthLabel }: { actions: MonitoringAction[]; dataQuality: DataQuality; monthLabel: string }) {
  const sorted = [...actions].sort((a, b) => ORDER[a.severity] - ORDER[b.severity]);
  const issues = [...dataQuality.errors, ...dataQuality.warnings];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alerts and recommended actions</CardTitle>
        <CardDescription>
          {monthLabel} · {sorted.length} raised · data-quality checks {dataQuality.ok ? 'passed' : 'failed'}
          {issues.length > 0 ? ` with ${issues.length} ${dataQuality.errors.length > 0 ? 'issue' : 'warning'}${issues.length === 1 ? '' : 's'}` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <CheckCircle2 className="size-4 text-ok-fg" aria-hidden="true" />
            No alerts were raised for this month.
          </p>
        ) : (
          <ul className="divide-y">
            {sorted.map((a) => (
              <li key={a.issue} className="grid gap-x-4 gap-y-1 py-3 first:pt-0 last:pb-0 sm:grid-cols-[6.5rem_minmax(0,1fr)]">
                <div>
                  <StatusBadge status={a.severity} />
                </div>
                <div className="space-y-1 text-sm">
                  <p className="font-medium">{a.issue}</p>
                  <p className="leading-relaxed text-muted-foreground">
                    <span className="font-medium text-foreground">Recommended action: </span>
                    {a.action}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
