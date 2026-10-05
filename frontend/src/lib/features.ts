/** Human-readable names for model features and fraud patterns returned by the API. */

const FEATURE_LABELS: Record<string, string> = {
  amount: 'Amount',
  log_amount: 'Amount (log)',
  amount_is_round: 'Round amount',
  hour: 'Hour of day',
  day_of_week: 'Day of week',
  is_night: 'Night-time payment',
  is_weekend: 'Weekend payment',
  hour_sin: 'Time of day (sin)',
  hour_cos: 'Time of day (cos)',
  dev_cnt_10m: 'Device payments, last 10 min',
  dev_cnt_1h: 'Device payments, last hour',
  dev_cnt_24h: 'Device payments, last 24 h',
  dev_cnt_7d: 'Device payments, last 7 days',
  dev_cnt_30d: 'Device payments, last 30 days',
  dev_amt_sum_1h: 'Device spend, last hour',
  dev_amt_sum_24h: 'Device spend, last 24 h',
  dev_secs_since_last: 'Seconds since device’s last payment',
  dev_age_days: 'Device age (days)',
  dev_logamt_z: 'Amount vs device’s usual (z-score)',
  dev_amt_ratio: 'Amount vs device’s 30-day average',
  dev_new_merchant: 'Merchant new for device',
  dev_new_merchants_24h: 'New merchants for device, 24 h',
  dev_new_service: 'Payment method new for device',
  dev_fail_cnt_1h: 'Device failed attempts, last hour',
  dev_fail_cnt_24h: 'Device failed attempts, 24 h',
  mer_cnt_1h: 'Merchant payments, last hour',
  mer_cnt_30d: 'Merchant payments, last 30 days',
  mer_logamt_z: 'Amount vs merchant’s usual (z-score)',
  mer_fraud_rate_lag: 'Merchant recent fraud rate',
  service_type: 'Payment method',
  merchant_type: 'Merchant risk tier',
  mcc_code: 'Merchant category (MCC)',
  issuer_bank: 'Issuer bank',
  merchant_state: 'Merchant state',
};

export function featureLabel(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature.replace(/_/g, ' ');
}

const PATTERN_LABELS: Record<string, string> = {
  A_takeover: 'A · Account takeover',
  B_mule: 'B · Mule',
  C_stealth: 'C · Stealth',
  D_upi_scam: 'D · UPI scam',
  genuine: 'Label noise (disputed genuine)',
};

export function patternLabel(group: string): string {
  return PATTERN_LABELS[group] ?? group.replace(/_/g, ' ');
}

/** Reason values are numbers for engineered features and strings for categorical ones. */
export function formatFeatureValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined) return '–';
  if (typeof value === 'string') return value;
  if (Number.isInteger(value)) return new Intl.NumberFormat('en-IN').format(value);
  return new Intl.NumberFormat('en-IN', Math.abs(value) >= 100 ? { maximumFractionDigits: 0 } : { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
}
