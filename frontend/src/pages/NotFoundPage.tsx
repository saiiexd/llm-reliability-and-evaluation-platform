import { Link } from "react-router-dom";
import { SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";

export function NotFoundPage() {
  return (
    <div>
      <SectionHeading eyebrow="404" title="Page Not Found" />
      <EmptyState
        title="Page not found"
        detail="The requested page route does not exist in the research console."
      />
      <div className="mt-6 text-center">
        <Link
          to="/"
          className="inline-block border border-line bg-paper px-4 py-2 font-mono text-xs uppercase tracking-wide hover:bg-paper-dim"
        >
          &larr; Return to Console Overview
        </Link>
      </div>
    </div>
  );
}

