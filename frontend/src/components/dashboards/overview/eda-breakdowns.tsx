'use client';

import { ChartCard, RateBarChart, type ChartTable, type RateDatum } from '@/components/charts';
import { formatHour, formatNumber, formatPercent } from '@/lib/format';
import type { Overview, RateRow } from '@/types';

function toData<T extends RateRow>(rows: T[], label: (row: T) => string): RateDatum[] {
  return rows.map((row) => ({ label: label(row), fraud_rate: row.fraud_rate, transactions: row.transactions, frauds: row.frauds }));
}

function toTable(heading: string, data: RateDatum[]): ChartTable {
  return {
    columns: [heading, 'Fraud rate', 'Transactions', 'Frauds'],
    rows: data.map((d) => [d.label, formatPercent(d.fraud_rate, 2), formatNumber(d.transactions), formatNumber(d.frauds)]),
  };
}

const TIER_ORDER = ['low', 'medium', 'high'];
const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const METHOD: Record<string, string> = { upi: 'UPI', imps: 'IMPS', wallet: 'Wallet', netbanking: 'Net banking' };

/** Fraud rate by segment over the development period. */
export function EdaBreakdowns({ overview }: { overview: Overview }) {
  const method = toData([...overview.by_service_type].sort((a, b) => b.fraud_rate - a.fraud_rate), (r) => METHOD[r.service_type] ?? r.service_type);
  const tier = toData(
    [...overview.by_merchant_type].sort((a, b) => TIER_ORDER.indexOf(a.merchant_type) - TIER_ORDER.indexOf(b.merchant_type)),
    (r) => titleCase(r.merchant_type)
  );
  const hour = toData(overview.by_hour, (r) => formatHour(r.hour));
  const amount = toData(overview.by_amount, (r) => `₹${r.amount}`);
  const velocity = toData(overview.by_device_velocity_1h, (r) => r.payments_last_hour);
  const relationship = toData(overview.by_new_merchant, (r) => titleCase(r.merchant_relationship));
  const mcc = toData([...overview.by_mcc].sort((a, b) => b.fraud_rate - a.fraud_rate), (r) => r.mcc_title);

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <ChartCard title="Fraud rate by payment method" height={200} table={toTable('Payment method', method)}>
        <RateBarChart data={method} showValues labelWidth={84} />
      </ChartCard>
      <ChartCard title="Fraud rate by merchant risk tier" height={200} table={toTable('Risk tier', tier)}>
        <RateBarChart data={tier} showValues labelWidth={64} />
      </ChartCard>
      <ChartCard
        title="Fraud rate by device velocity"
        description="Other payments from the same device in the previous hour."
        height={200}
        table={toTable('Payments in last hour', velocity)}
      >
        <RateBarChart data={velocity} orientation="columns" showValues />
      </ChartCard>
      <ChartCard title="Fraud rate by amount band" height={220} table={toTable('Amount', amount)}>
        <RateBarChart data={amount} orientation="columns" showValues />
      </ChartCard>
      <ChartCard title="Fraud rate by hour of day" height={220} table={toTable('Hour', hour)} className="md:col-span-2">
        <RateBarChart data={hour} orientation="columns" />
      </ChartCard>
      <ChartCard title="Fraud rate by merchant category (MCC)" height={Math.max(260, mcc.length * 24)} table={toTable('Category', mcc)} className="md:col-span-2">
        <RateBarChart data={mcc} labelWidth={224} />
      </ChartCard>
      <ChartCard
        title="Fraud rate by merchant relationship"
        description="Whether the device has paid this merchant before."
        height={200}
        table={toTable('Relationship', relationship)}
      >
        <RateBarChart data={relationship} showValues labelWidth={108} />
      </ChartCard>
    </div>
  );
}
