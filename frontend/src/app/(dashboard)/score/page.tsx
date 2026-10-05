import type { Metadata } from 'next';
import { ScoreDashboard } from '@/components/dashboards/score';

export const metadata: Metadata = { title: 'Score a transaction' };

export default function ScorePage() {
  return <ScoreDashboard />;
}
