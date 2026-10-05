import { AlertTriangle, Ban, CheckCircle2, Eye, FileText, OctagonAlert, Sparkles } from 'lucide-react';
import { Badge, type BadgeVariant } from '@/components/ui/badge';
import type { Band, Status } from '@/types';

/** Status and band are always shown as icon + text, never as colour alone. */

const STATUS: Record<Status, { label: string; variant: BadgeVariant; Icon: typeof CheckCircle2 }> = {
  ok: { label: 'OK', variant: 'ok', Icon: CheckCircle2 },
  warn: { label: 'Warning', variant: 'warn', Icon: AlertTriangle },
  alert: { label: 'Alert', variant: 'alert', Icon: OctagonAlert },
};

export function StatusBadge({ status, label, className }: { status: Status; label?: string; className?: string }) {
  const { label: fallback, variant, Icon } = STATUS[status] ?? STATUS.warn;
  return (
    <Badge variant={variant} className={className}>
      <Icon aria-hidden="true" />
      {label ?? fallback}
    </Badge>
  );
}

const BAND: Record<Band, { label: string; variant: BadgeVariant; Icon: typeof CheckCircle2 }> = {
  approve: { label: 'Approve', variant: 'ok', Icon: CheckCircle2 },
  review: { label: 'Manual review', variant: 'warn', Icon: Eye },
  block: { label: 'Block', variant: 'alert', Icon: Ban },
};

export function BandBadge({ band, className }: { band: Band; className?: string }) {
  const config = BAND[band];
  if (!config) return <Badge className={className}>{band}</Badge>;
  const { label, variant, Icon } = config;
  return (
    <Badge variant={variant} className={className}>
      <Icon aria-hidden="true" />
      {label}
    </Badge>
  );
}

export function SourceBadge({ source, model }: { source: string; model?: string | null }) {
  const isLlm = source === 'llm';
  const Icon = isLlm ? Sparkles : FileText;
  return (
    <Badge variant="outline">
      <Icon aria-hidden="true" />
      {isLlm ? `LLM${model ? ` · ${model}` : ''}` : source === 'template' ? 'Template' : source}
    </Badge>
  );
}
