/** Display formatting. Money is INR with Indian digit grouping; rates are percentages. */

const DASH = '–';
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const inr0 = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const inr2 = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const inrCompact = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', notation: 'compact', maximumFractionDigits: 2 });
const int = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });

function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** ₹10,297.82 — exact amount of a single payment. */
export function formatINR(value: number | null | undefined, opts: { decimals?: boolean } = {}): string {
  if (!isNum(value)) return DASH;
  return (opts.decimals ? inr2 : inr0).format(value);
}

/** ₹1.04Cr / ₹23.86L — aggregate amounts. */
export function formatINRCompact(value: number | null | undefined): string {
  if (!isNum(value)) return DASH;
  return Math.abs(value) < 100_000 ? inr0.format(value) : inrCompact.format(value);
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (!isNum(value)) return DASH;
  return digits === 0
    ? int.format(value)
    : new Intl.NumberFormat('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

/** 0.0112 -> "1.12%". */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (!isNum(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Signed percentage-point difference between two rates: "+1.2 pp". */
export function formatPointDelta(value: number, baseline: number, digits = 1): string {
  const d = (value - baseline) * 100;
  const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
  return `${sign}${Math.abs(d).toFixed(digits)} pp`;
}

/** Unitless statistic such as PR-AUC, PSI or KS. */
export function formatStat(value: number | null | undefined, digits = 3): string {
  if (!isNum(value)) return DASH;
  return value.toFixed(digits);
}

/** "2026-07" -> "Jul 2026". */
export function formatMonth(month: string): string {
  const [y, m] = month.split('-').map(Number);
  if (!y || !m || m > 12) return month;
  return `${MONTHS[m - 1]} ${y}`;
}

/** "2026-06-08" -> "8 Jun 2026". */
export function formatDate(date: string | null | undefined): string {
  if (!date) return DASH;
  const [y, m, d] = date.slice(0, 10).split('-').map(Number);
  if (!y || !m || !d || m > 12) return date;
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

/** "2026-09-30 03:15:43" -> "30 Sep 2026, 03:15". The API sends naive timestamps; shown as-is. */
export function formatDateTime(ts: string | null | undefined): string {
  if (!ts) return DASH;
  const [date, time = ''] = ts.replace('T', ' ').split(' ');
  return `${formatDate(date)}${time ? `, ${time.slice(0, 5)}` : ''}`;
}

export function formatHour(hour: number): string {
  return `${String(hour).padStart(2, '0')}:00`;
}

export { DASH };
