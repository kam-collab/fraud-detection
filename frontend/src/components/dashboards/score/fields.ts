import type { Transaction } from '@/types';

export type FieldName = keyof Transaction;
export type FormValues = Record<FieldName, string>;

export interface FieldConfig {
  name: FieldName;
  label: string;
  required?: boolean;
  type?: 'text' | 'number';
  placeholder?: string;
  hint?: string;
  options?: { value: string; label: string }[];
}

/** The assignment's transaction schema, in form order. */
export const FIELDS: FieldConfig[] = [
  { name: 'amount', label: 'Amount (INR)', required: true, type: 'number', placeholder: '850.00' },
  {
    name: 'service_type',
    label: 'Payment method',
    required: true,
    options: [
      { value: 'upi', label: 'UPI' },
      { value: 'wallet', label: 'Wallet' },
      { value: 'imps', label: 'IMPS' },
      { value: 'netbanking', label: 'Net banking' },
    ],
  },
  { name: 'device_id', label: 'Device ID', required: true, placeholder: 'DEV0001234' },
  { name: 'merchant_id', label: 'Merchant ID', required: true, placeholder: 'M100101' },
  {
    name: 'merchant_type',
    label: 'Merchant risk tier',
    required: true,
    options: [
      { value: 'low', label: 'Low' },
      { value: 'medium', label: 'Medium' },
      { value: 'high', label: 'High' },
    ],
  },
  { name: 'mcc_code', label: 'MCC code', required: true, type: 'number', placeholder: '5411' },
  { name: 'mcc_title', label: 'MCC title', placeholder: 'Grocery Stores' },
  { name: 'issuer_bank', label: 'Issuer bank', placeholder: 'HDFC' },
  { name: 'merchant_state', label: 'Merchant state', placeholder: 'Karnataka' },
  { name: 'merchant_city', label: 'Merchant city', placeholder: 'Bengaluru' },
  { name: 'currency_code', label: 'Currency', placeholder: 'INR' },
  { name: 'request_time', label: 'Request time', placeholder: '2026-09-02 11:36:34', hint: 'Optional. Defaults to the service clock.' },
  { name: 'request_id', label: 'Request ID', placeholder: 'TXN…', hint: 'Optional. Generated when empty.' },
  { name: 'request_status', label: 'Request status', placeholder: 'SUCCESS', hint: 'Optional. Not known at decision time.' },
];

export const EMPTY_FORM: FormValues = {
  request_id: '',
  request_time: '',
  service_type: 'upi',
  device_id: '',
  merchant_id: '',
  merchant_state: '',
  merchant_city: '',
  merchant_type: 'low',
  mcc_code: '',
  mcc_title: '',
  issuer_bank: '',
  currency_code: 'INR',
  amount: '',
  request_status: '',
};

export function fromTransaction(txn: Transaction): FormValues {
  const values = { ...EMPTY_FORM };
  for (const { name } of FIELDS) {
    const v = txn[name];
    values[name] = v === null || v === undefined ? '' : String(v);
  }
  return values;
}

const NUMERIC: FieldName[] = ['amount', 'mcc_code'];

/**
 * Builds the request body. Empty fields are omitted and numbers are parsed, but
 * nothing is rejected here: the API is the single source of validation, and its
 * HTTP 422 response is shown next to the form.
 */
export function toPayload(values: FormValues): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const { name } of FIELDS) {
    const raw = values[name].trim();
    if (raw === '') continue;
    if (NUMERIC.includes(name)) {
      const n = Number(raw);
      body[name] = Number.isFinite(n) ? n : raw;
    } else {
      body[name] = raw;
    }
  }
  return body;
}
