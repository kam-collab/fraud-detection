/**
 * Shared types for the fraud-detection API.
 * Shapes were taken from real responses of the running service, not guessed.
 */

export type Band = 'approve' | 'review' | 'block';
export type Status = 'ok' | 'warn' | 'alert';
export type SampleKind = 'random' | 'fraud' | 'genuine' | 'review' | 'blocked';
export type ReviewDecisionValue = 'approve' | 'decline';
export type DateRange = [string, string];
export type Interval = [number, number];

// ---------------------------------------------------------------- health
export interface Health {
  status: string;
}

export interface Readiness {
  status: string;
  model_version: string;
  history_rows: number;
}

// -------------------------------------------------------------- overview
export interface RateRow {
  transactions: number;
  frauds: number;
  fraud_rate: number;
}

export type RateBreakdown<K extends string, V = string> = RateRow & Record<K, V>;

export interface Overview {
  period: DateRange;
  transactions: number;
  frauds: number;
  fraud_rate: number;
  fraud_amount_share: number;
  monthly: RateBreakdown<'month'>[];
  by_service_type: RateBreakdown<'service_type'>[];
  by_merchant_type: RateBreakdown<'merchant_type'>[];
  by_request_status: RateBreakdown<'request_status'>[];
  by_mcc: RateBreakdown<'mcc_title'>[];
  by_hour: RateBreakdown<'hour', number>[];
  by_amount: RateBreakdown<'amount'>[];
  by_device_velocity_1h: RateBreakdown<'payments_last_hour'>[];
  by_new_merchant: RateBreakdown<'merchant_relationship'>[];
}

// ----------------------------------------------------------------- model
export interface ThresholdMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  false_positive_rate: number;
  flag_rate: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface EvalMetrics extends ThresholdMetrics {
  pr_auc: number;
  roc_auc: number;
  brier: number;
  n: number;
  fraud_rate: number;
}

export interface MetricIntervals {
  pr_auc: Interval;
  roc_auc: Interval;
  precision: Interval;
  recall: Interval;
  f1: Interval;
}

export interface BusinessMetrics {
  transactions: number;
  approval_rate: number;
  review_queue: number;
  review_rate: number;
  blocked: number;
  false_declines: number;
  false_decline_amount: number;
  genuine_sent_to_review: number;
  fraud_amount_total: number;
  fraud_loss_amount: number;
  fraud_amount_prevented: number;
  fraud_value_recall: number;
  total_cost: number;
  cost_without_model: number;
  /** Present in the monitoring report, absent from the training report. */
  false_decline_rate?: number;
  /** Present in the monitoring report, absent from the training report. */
  review_queue_per_day?: number;
}

export interface PrCurvePoint {
  threshold: number;
  precision: number;
  recall: number;
}

export interface PatternRecall {
  group: string;
  frauds: number;
  recall: number;
}

export interface ModelEvaluation {
  name: string;
  review_threshold: number;
  block_threshold: number;
  valid: EvalMetrics;
  test: EvalMetrics;
  test_ci95: MetricIntervals;
  test_block_band: ThresholdMetrics;
  'test_default_0.5': ThresholdMetrics;
  test_max_f1_threshold: ThresholdMetrics;
  test_business: BusinessMetrics;
  test_pr_curve: PrCurvePoint[];
  test_brier_uncalibrated: number;
  test_recall_by_pattern: PatternRecall[];
}

export interface DataWindows {
  train: DateRange;
  valid: DateRange;
  test: DateRange;
  label_delay_days: number;
}

export interface ModelCard {
  model_version: string;
  trained_at: string;
  data: { source: string; seed: number; rows: number; fraud_rate: number };
  windows: DataWindows;
  policy: { review_min_precision: number; block_min_precision: number };
  features: string[];
  algorithm: string;
  thresholds: { review: number; block: number };
  release_baseline: Record<string, number>;
  release_baseline_ci95: MetricIntervals;
  rollback: string;
}

export interface SelectionRow {
  model: string;
  valid_pr_auc: number;
  valid_roc_auc: number;
  valid_brier_uncalibrated: number;
}

export interface SplitRows {
  n: number;
  frauds: number;
  fraud_rate: number;
}

export interface ModelReport {
  /** Empty object when the model card file is missing on the server. */
  card: Partial<ModelCard>;
  champion: ModelEvaluation;
  baseline: ModelEvaluation;
  selection_table: SelectionRow[];
  rows: Record<'train' | 'valid' | 'test' | 'live', SplitRows>;
  windows: DataWindows;
}

// ------------------------------------------------------------ monitoring
export interface DriftRow {
  feature: string;
  kind: 'numeric' | 'categorical';
  psi: number;
  /** null for categorical features. */
  ks: number | null;
  missing_ref: number;
  missing_cur: number;
  status: Status;
}

export interface DataQuality {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

export interface PredictionDrift {
  score_psi: number;
  mean_score: number;
  baseline_mean_score: number;
  /** Share of payments at or above the review threshold (sent to review or blocked). */
  flag_rate: number;
  baseline_flag_rate: number;
  /** Relative change of flag_rate against the baseline: 0.56 = +56%. */
  flag_rate_change: number;
  block_rate: number;
  baseline_block_rate: number;
  /** Reflects both the score PSI and the flag-rate change. */
  status: Status;
}

export interface MonthPerformance {
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  fraud_rate: number;
  baseline_pr_auc: number;
  baseline_roc_auc: number;
  baseline_precision: number;
  baseline_recall: number;
  baseline_f1: number;
  pr_auc_status: Status;
  precision_status: Status;
  recall_status: Status;
  f1_status: Status;
  labels_available_from: string;
}

export interface MonthBusiness extends BusinessMetrics {
  false_decline_rate: number;
  baseline_approval_rate: number;
  baseline_review_rate: number;
  baseline_false_decline_rate: number;
  baseline_fraud_value_recall: number;
  review_queue_per_day: number;
}

export interface MonitoringAction {
  severity: Status;
  issue: string;
  action: string;
}

export interface MonitoringMonth {
  month: string;
  rows: number;
  status: Status;
  data_quality: DataQuality;
  data_drift: DriftRow[];
  prediction_drift: PredictionDrift;
  performance: MonthPerformance;
  business: MonthBusiness;
  recall_by_pattern: PatternRecall[];
  actions: MonitoringAction[];
}

/** `edges` has one more entry than each series; other keys are `baseline_june` and one per month. */
export type ScoreHistogram = { edges: number[]; baseline_june: number[] } & Record<string, number[]>;

export interface RemediationRow {
  model: string;
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  fp: number;
  fn: number;
  review_threshold: number;
  block_threshold: number;
  false_declines: number;
  review_queue: number;
  fraud_loss_amount: number;
  total_cost: number;
}

export interface MonitoringReport {
  model_version: string;
  reference: { inputs: DateRange; scores_and_metrics: DateRange };
  thresholds: { psi_warn: number; psi_alert: number; ks_warn: number; ks_alert: number };
  baseline: { metrics: EvalMetrics; ci95: MetricIntervals; business: BusinessMetrics };
  months: MonitoringMonth[];
  score_histogram: ScoreHistogram;
  remediation: { window: DateRange; rows: RemediationRow[] };
}

// --------------------------------------------------------------- scoring
/** Request body of POST /score. Sample transactions come back in the same shape. */
export interface Transaction {
  request_id?: string | null;
  request_time?: string | null;
  service_type: string;
  device_id: string;
  merchant_id: string;
  merchant_state?: string | null;
  merchant_city?: string | null;
  merchant_type: string;
  mcc_code: number;
  mcc_title?: string | null;
  issuer_bank?: string | null;
  currency_code?: string;
  amount: number;
  request_status?: string | null;
}

export interface Reason {
  feature: string;
  value: number | string | null;
  typical: number | string | null;
  log_odds_contribution: number;
}

export interface Explanation {
  text: string;
  source: 'template' | 'llm' | (string & {});
  model: string | null;
}

export interface ScoreResult {
  request_id: string;
  score: number;
  band: Band;
  thresholds: { review: number; block: number };
  model_version: string;
  reasons: Reason[];
  explanation: Explanation;
  warnings: string[];
}

export interface ExplainResult {
  request_id: string;
  score: number;
  band: Band;
  reasons: Reason[];
  reason_phrases: string[];
  explanation: Explanation;
}

// ---------------------------------------------------------- review queue
export interface ReviewItem {
  request_id: string;
  request_time: string;
  service_type: string;
  merchant_state: string | null;
  merchant_city: string | null;
  merchant_type: string;
  mcc_code: number;
  mcc_title: string | null;
  issuer_bank: string | null;
  currency_code: string;
  amount: number;
  score: number;
  band: Band;
  decision: ReviewDecisionValue | null;
  /** 1 = fraud, 0 = genuine. Only revealed once a decision has been recorded. */
  actual_label: 0 | 1 | null;
}

export interface ReviewQueue {
  total: number;
  pending: number;
  since: string;
  items: ReviewItem[];
}

export interface ReviewDecisionResult {
  request_id: string;
  decision: ReviewDecisionValue;
  actual_label: 0 | 1;
  correct: boolean;
}

// ---------------------------------------------------------------- errors
/** FastAPI/pydantic validation entry (HTTP 422, `detail` is an array). */
export interface FieldValidationError {
  type: string;
  loc: (string | number)[];
  msg: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

/** Domain validation failure from the scoring service (HTTP 422, `detail` is an object). */
export interface DomainValidationDetail {
  errors: string[];
  warnings: string[];
}
