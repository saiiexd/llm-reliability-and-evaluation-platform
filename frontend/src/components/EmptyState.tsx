/**
 * An honest "nothing here" state. Used instead of a fabricated chart or
 * zero value whenever there is genuinely no data to show.
 */
export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-dashed border-line px-5 py-8 text-center text-sm">
      <p className="font-medium text-ink">{title}</p>
      {detail && <p className="mt-1 text-ink-soft">{detail}</p>}
    </div>
  );
}
