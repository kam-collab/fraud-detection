'use client';

import { RefreshCw, ServerCrash, TriangleAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { API_BASE_URL, isApiError, isNetworkError } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
  title?: string;
  className?: string;
}

function Code({ children }: { children: string }) {
  return <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-foreground">{children}</code>;
}

/** Explains what failed and what to do next; distinguishes "API down" from an HTTP error. */
export function ErrorState({ error, onRetry, title, className }: ErrorStateProps) {
  const offline = isNetworkError(error);
  const Icon = offline ? ServerCrash : TriangleAlert;
  const status = isApiError(error) && error.status > 0 ? error.status : null;
  const message = error instanceof Error ? error.message : 'Unknown error';

  return (
    <div role="alert" className={cn('flex flex-col items-start gap-3 rounded-card border border-alert-fg/30 bg-alert-bg p-5', className)}>
      <div className="flex items-center gap-2 text-alert-fg">
        <Icon className="size-5 shrink-0" aria-hidden="true" />
        <p className="text-sm font-semibold">{offline ? 'API not reachable' : (title ?? 'Could not load this data')}</p>
      </div>
      {offline ? (
        <p className="text-sm leading-relaxed text-foreground">
          No response from <Code>{API_BASE_URL}</Code>. Start it with <Code>make api</Code> from the project root, then retry. If
          the API runs elsewhere, set <Code>NEXT_PUBLIC_API_URL</Code>.
        </p>
      ) : (
        <p className="text-sm leading-relaxed text-foreground">
          {status ? `HTTP ${status}: ` : ''}
          {message}
        </p>
      )}
      {onRetry ? (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw aria-hidden="true" />
          Retry
        </Button>
      ) : null}
    </div>
  );
}
