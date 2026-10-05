import type { Metadata } from 'next';
import { ReviewDashboard } from '@/components/dashboards/review';

export const metadata: Metadata = { title: 'Review queue' };

export default function ReviewPage() {
  return <ReviewDashboard />;
}
