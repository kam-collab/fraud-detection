'use client';

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatDateTime, formatINR, formatNumber, formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { ReviewItem } from '@/types';
import { OutcomeBadge } from './outcome';

interface QueueTableProps {
  items: ReviewItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function QueueTable({ items, selectedId, onSelect }: QueueTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Request</TableHead>
          <TableHead numeric>Amount</TableHead>
          <TableHead>Merchant</TableHead>
          <TableHead numeric>Score</TableHead>
          <TableHead>Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => {
          const selected = item.request_id === selectedId;
          return (
            <TableRow key={item.request_id} aria-selected={selected} className={cn('transition-colors', selected ? 'bg-accent' : 'hover:bg-muted/60')}>
              <TableCell>
                <button
                  type="button"
                  onClick={() => onSelect(item.request_id)}
                  aria-pressed={selected}
                  aria-label={`Open ${item.request_id}`}
                  className="rounded font-mono text-xs font-semibold text-primary underline-offset-4 hover:underline"
                >
                  {item.request_id}
                </button>
                <span className="block whitespace-nowrap text-xs text-subtle">{formatDateTime(item.request_time)}</span>
              </TableCell>
              <TableCell numeric className="whitespace-nowrap">
                <span className="block font-medium">{formatINR(item.amount)}</span>
                <span className="text-xs uppercase text-subtle">{item.service_type}</span>
              </TableCell>
              <TableCell>
                <span className="block max-w-44 truncate" title={item.mcc_title ?? undefined}>
                  {item.mcc_title ?? '–'}
                </span>
                <span className="text-xs text-subtle">
                  {item.merchant_city ?? 'Unknown city'} · {item.merchant_type} risk
                </span>
              </TableCell>
              <TableCell numeric>{formatPercent(item.score, 1)}</TableCell>
              <TableCell>
                <OutcomeBadge decision={item.decision} actualLabel={item.actual_label} />
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

interface PaginationProps {
  total: number;
  limit: number;
  offset: number;
  onOffsetChange: (offset: number) => void;
  isFetching: boolean;
}

export function Pagination({ total, limit, offset, onOffsetChange, isFetching }: PaginationProps) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);

  return (
    <nav aria-label="Review queue pages" className="flex flex-wrap items-center justify-between gap-3 px-3 pt-3 text-sm">
      <p className="text-muted-foreground" aria-live="polite">
        {formatNumber(from)}–{formatNumber(to)} of {formatNumber(total)}
        {isFetching ? ' · updating…' : ''}
      </p>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => onOffsetChange(Math.max(0, offset - limit))} disabled={offset === 0}>
          <ChevronLeft aria-hidden="true" />
          Previous
        </Button>
        <span className="tabular text-muted-foreground">
          Page {page} of {pages}
        </span>
        <Button variant="outline" size="sm" onClick={() => onOffsetChange(offset + limit)} disabled={offset + limit >= total}>
          Next
          <ChevronRight aria-hidden="true" />
        </Button>
      </div>
    </nav>
  );
}
