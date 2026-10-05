import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium [&_svg]:size-3.5 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        neutral: 'border-border bg-muted text-muted-foreground',
        outline: 'border-input text-foreground',
        ok: 'border-ok-fg/30 bg-ok-bg text-ok-fg',
        warn: 'border-warn-fg/30 bg-warn-bg text-warn-fg',
        alert: 'border-alert-fg/30 bg-alert-bg text-alert-fg',
      },
    },
    defaultVariants: { variant: 'neutral' },
  }
);

export type BadgeVariant = NonNullable<VariantProps<typeof badgeVariants>['variant']>;

function Badge({ className, variant, ...props }: React.ComponentProps<'span'> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
