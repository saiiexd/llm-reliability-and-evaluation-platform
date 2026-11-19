import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`border border-line bg-paper p-4 ${className}`}>{children}</div>;
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-end justify-between border-b border-line pb-2">
      <div>
        {eyebrow && (
          <p className="font-mono text-xs tracking-wide text-ink-soft uppercase">{eyebrow}</p>
        )}
        <h2 className="font-serif text-xl">{title}</h2>
        {description && <p className="mt-1 text-xs text-ink-soft">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function Stat({
  label,
  value,
  subtext,
}: {
  label: string;
  value: ReactNode;
  subtext?: ReactNode;
}) {
  return (
    <div className="border border-line px-4 py-3">
      <p className="font-mono text-xs tracking-wide text-ink-soft uppercase">{label}</p>
      <p className="font-serif text-2xl">{value}</p>
      {subtext && <p className="mt-1 text-[11px] text-ink-soft">{subtext}</p>}
    </div>
  );
}
