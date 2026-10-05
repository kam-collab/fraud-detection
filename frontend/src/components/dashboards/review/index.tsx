'use client';

import { MousePointerClick } from 'lucide-react';
import { useState } from 'react';
import { EmptyState } from '@/components/common/empty-state';
import { ErrorState } from '@/components/common/error-state';
import { KpiTile } from '@/components/common/kpi-tile';
import { LoadingState } from '@/components/common/loading-state';
import { PageHeader } from '@/components/common/page-header';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useModelReport, useReviewQueue } from '@/hooks';
import { formatDateTime, formatNumber } from '@/lib/format';
import { DetailPanel } from './detail-panel';
import { Pagination, QueueTable } from './queue-table';

const PAGE_SIZES = [10, 25, 50];

export function ReviewDashboard() {
  const [limit, setLimit] = useState(10);
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const queue = useReviewQueue(limit, offset);
  // Only used to draw the thresholds on the score meter; the page works without it.
  const model = useModelReport();

  const header = (
    <PageHeader
      title="Review queue"
      description="Transactions the model sent to manual review, highest score first. Open one to see why it was flagged, then approve or decline it."
    />
  );

  if (queue.isPending) {
    return (
      <>
        {header}
        <LoadingState label="Loading review queue" blocks={2} />
      </>
    );
  }
  if (queue.isError) {
    return (
      <>
        {header}
        <ErrorState error={queue.error} onRetry={() => queue.refetch()} title="Could not load the review queue" />
      </>
    );
  }

  const { total, pending, since, items } = queue.data;
  const selected = items.find((item) => item.request_id === selectedId) ?? null;
  const thresholds = model.data ? { review: model.data.champion.review_threshold, block: model.data.champion.block_threshold } : undefined;

  const changePage = (next: number) => {
    setOffset(next);
    setSelectedId(null);
  };

  return (
    <>
      {header}
      <section aria-label="Queue summary" className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <KpiTile label="In queue" value={formatNumber(total)} hint={`Flagged since ${formatDateTime(since)}`} />
        <KpiTile label="Pending" value={formatNumber(pending)} hint="Awaiting a decision" />
        <KpiTile label="Decided" value={formatNumber(total - pending)} hint="This API session" className="col-span-2 sm:col-span-1" />
      </section>

      {total === 0 ? (
        <EmptyState title="The review queue is empty" description="No transactions fell in the review band in the most recent days of data." />
      ) : (
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_26rem]">
          <Card className="min-w-0">
            <CardHeader className="flex-row flex-wrap items-end justify-between gap-3">
              <div className="space-y-1">
                <CardTitle>Flagged transactions</CardTitle>
                <CardDescription>Select a request ID to open it.</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="page-size">Rows per page</Label>
                <Select
                  id="page-size"
                  className="w-20"
                  value={limit}
                  onChange={(e) => {
                    setLimit(Number(e.target.value));
                    changePage(0);
                  }}
                >
                  {PAGE_SIZES.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </Select>
              </div>
            </CardHeader>
            <CardContent className="px-2 pt-3">
              {items.length === 0 ? (
                <EmptyState title="No transactions on this page" description="Go back to an earlier page." />
              ) : (
                <QueueTable items={items} selectedId={selectedId} onSelect={setSelectedId} />
              )}
              <Pagination total={total} limit={limit} offset={offset} onOffsetChange={changePage} isFetching={queue.isFetching} />
            </CardContent>
          </Card>

          <div className="min-w-0 xl:sticky xl:top-[4.5rem]">
            {selected ? (
              // Keyed so the decision state of one transaction never leaks into the next.
              <DetailPanel key={selected.request_id} item={selected} thresholds={thresholds} />
            ) : (
              <EmptyState
                icon={<MousePointerClick />}
                title="No transaction selected"
                description="Select a request ID in the table to see its reasons and explanation."
                className="min-h-64"
              />
            )}
          </div>
        </div>
      )}
    </>
  );
}
