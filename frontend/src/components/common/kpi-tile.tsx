import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface KpiTileProps {
  label: string;
  value: string;
  /** Context line: period, confidence interval, comparison. */
  hint?: ReactNode;
  /** Optional badge or delta shown beside the label. */
  aside?: ReactNode;
  className?: string;
}

export function KpiTile({ label, value, hint, aside, className }: KpiTileProps) {
  return (
    <div className={cn('flex flex-col gap-1.5 rounded-card border bg-card p-4', className)}>
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        {aside}
      </div>
      <p className="text-2xl font-semibold tracking-tight">{value}</p>
      {hint ? <p className="text-xs leading-relaxed text-subtle">{hint}</p> : null}
    </div>
  );
}
