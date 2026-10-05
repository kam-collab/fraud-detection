import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        outline: 'border border-input bg-card text-foreground hover:bg-muted',
        ghost: 'text-foreground hover:bg-muted',
        approve: 'border border-ok-fg/40 bg-ok-bg text-ok-fg hover:bg-ok-bg/70',
        decline: 'border border-alert-fg/40 bg-alert-bg text-alert-fg hover:bg-alert-bg/70',
      },
      size: {
        default: 'h-9 px-4',
        sm: 'h-8 px-3 text-xs',
        icon: 'size-9',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  }
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

function Button({ className, variant, size, asChild = false, type, ...props }: ButtonProps) {
  const Comp = asChild ? Slot : 'button';
  return <Comp className={cn(buttonVariants({ variant, size }), className)} type={asChild ? undefined : (type ?? 'button')} {...props} />;
}

export { Button, buttonVariants };
