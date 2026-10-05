import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 space-y-1">
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">{title}</h1>
        {description ? <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function SectionHeading({ title, description }: { title: string; description?: ReactNode }) {
  return (
    <div className="space-y-1">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      {description ? <p className="max-w-3xl text-sm text-muted-foreground">{description}</p> : null}
    </div>
  );
}
