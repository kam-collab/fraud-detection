import { Inbox } from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  title: string;
  description?: ReactNode;
  icon?: ReactNode;
  className?: string;
}

export function EmptyState({ title, description, icon, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-2 rounded-card border border-dashed p-8 text-center', className)}>
      <span className="text-subtle [&_svg]:size-6" aria-hidden="true">
        {icon ?? <Inbox />}
      </span>
      <p className="text-sm font-medium">{title}</p>
      {description ? <p className="max-w-md text-sm text-muted-foreground">{description}</p> : null}
    </div>
  );
}
