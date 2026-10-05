import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

/** Skeleton placeholder with a screen-reader announcement. */
export function LoadingState({ label = 'Loading', tiles = 0, blocks = 2, className }: { label?: string; tiles?: number; blocks?: number; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={cn('space-y-4', className)}>
      <span className="sr-only">{label}…</span>
      {tiles > 0 ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {Array.from({ length: tiles }, (_, i) => (
            <Skeleton key={i} className="h-24 rounded-card" />
          ))}
        </div>
      ) : null}
      {Array.from({ length: blocks }, (_, i) => (
        <Skeleton key={i} className="h-56 rounded-card" />
      ))}
    </div>
  );
}
