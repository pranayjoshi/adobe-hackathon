import type { ReactNode } from "react";

export const ROLE_COLORS: Record<string, string> = {
  headline: "#34d399",
  subhead: "#5eead4",
  body: "#67e8f9",
  legal: "#94a3b8",
  hero: "#f472b6",
  image: "#f9a8d4",
  logo: "#fbbf24",
  cta: "#fb923c",
  badge: "#a78bfa",
  badges: "#a78bfa",
  sticker: "#f87171",
  accent: "#c4b5fd",
  group: "#60a5fa",
  decoration: "#71717a",
  background: "#52525b",
};

export function roleColor(role?: string) {
  return (role && ROLE_COLORS[role]) || "#a1a1aa";
}

export function scoreColor(v: number) {
  if (v >= 85) return "#34d399";
  if (v >= 70) return "#a3e635";
  if (v >= 55) return "#fbbf24";
  return "#f87171";
}

export function ScoreRing({ value, size = 44, stroke = 4 }: { value: number; size?: number; stroke?: number }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const color = scoreColor(value);
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,.08)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - Math.max(0, Math.min(100, value)) / 100)}
          style={{ transition: "stroke-dashoffset .8s cubic-bezier(.2,.7,.2,1)" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-[13px] font-semibold tabular-nums" style={{ color }}>
        {Math.round(value)}
      </div>
    </div>
  );
}

export function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-[78px] shrink-0 text-zinc-500">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.max(2, value)}%`, background: scoreColor(value), transition: "width .8s cubic-bezier(.2,.7,.2,1)" }}
        />
      </div>
      <span className="w-7 text-right tabular-nums text-zinc-400">{Math.round(value)}</span>
    </div>
  );
}

export function Icon({ name, className = "h-4 w-4" }: { name: string; className?: string }) {
  const p: Record<string, ReactNode> = {
    spark: <path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4L12 3z" />,
    upload: <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" />,
    download: <path d="M12 4v12m0 0l-4-4m4 4l4-4M4 18v1a1 1 0 001 1h14a1 1 0 001-1v-1" />,
    check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
    x: <path d="M6 6l12 12M18 6L6 18" />,
    warn: <path d="M12 9v4m0 3.5v.01M10.3 4.3L2.6 18a2 2 0 001.7 3h15.4a2 2 0 001.7-3L13.7 4.3a2 2 0 00-3.4 0z" />,
    info: <path d="M12 16v-4m0-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />,
    layers: <path d="M12 3l9 5-9 5-9-5 9-5zm-9 9l9 5 9-5M3 16l9 5 9-5" />,
    plus: <path d="M12 5v14M5 12h14" />,
    eye: <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zm10 3a3 3 0 100-6 3 3 0 000 6z" />,
    compare: <path d="M12 3v18M5 7h4v10H5zM15 5h4v14h-4z" />,
    copy: <path d="M8 8h11v11H8zM5 16V5h11" />,
    expand: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
  };
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className={className}>
      {p[name]}
    </svg>
  );
}

export function Toggle({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      onClick={() => onChange(!on)}
      className={`inline-flex items-center gap-2 rounded-lg px-2 py-1 text-xs transition ${on ? "text-emerald-300" : "text-zinc-400 hover:text-zinc-200"}`}
    >
      <span className={`relative h-4 w-7 rounded-full transition ${on ? "bg-emerald-400/30" : "bg-white/10"}`}>
        <span className={`absolute top-0.5 h-3 w-3 rounded-full transition-all ${on ? "left-3.5 bg-emerald-300" : "left-0.5 bg-zinc-400"}`} />
      </span>
      {label}
    </button>
  );
}

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative h-8 w-8">
        <div className="absolute inset-0 rounded-[10px] bg-gradient-to-br from-emerald-300 via-teal-300 to-violet-400" />
        <div className="absolute inset-[5px] rounded-[6px] border-2 border-ink-950/80" />
        <div className="absolute bottom-[9px] right-[9px] h-2 w-3.5 rounded-sm bg-ink-950/80" />
      </div>
      <div className="leading-none">
        <div className="font-display text-[22px] tracking-tight text-white">Chameleon</div>
        <div className="mt-0.5 text-[10px] uppercase tracking-[0.18em] text-zinc-500">Design retargeting</div>
      </div>
    </div>
  );
}
