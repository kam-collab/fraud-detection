import * as React from 'react';
import { cn } from '@/lib/utils';

function Card({ className, ...props }: React.ComponentProps<'section'>) {
  return <section className={cn('rounded-card border bg-card text-foreground', className)} {...props} />;
}

function CardHeader({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('flex flex-col gap-1 px-5 pt-5', className)} {...props} />;
}

function CardTitle({ className, ...props }: React.ComponentProps<'h2'>) {
  return <h2 className={cn('text-sm font-semibold leading-snug', className)} {...props} />;
}

function CardDescription({ className, ...props }: React.ComponentProps<'p'>) {
  return <p className={cn('text-xs leading-relaxed text-muted-foreground', className)} {...props} />;
}

function CardContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('p-5', className)} {...props} />;
}

export { Card, CardHeader, CardTitle, CardDescription, CardContent };
