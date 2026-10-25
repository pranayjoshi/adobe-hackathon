import { useState } from "react";
import type { Box } from "../types";
import { roleColor } from "./ui";

type Props = {
  src: string | null;
  width: number;
  height: number;
  overlay?: (Box & { id: string })[];
  roleOf?: (id: string) => string | undefined;
  labelOf?: (id: string) => string | undefined;
  showOverlay?: boolean;
  className?: string;
  maxHeight?: number;
};

/** A rendered design at its true aspect ratio, with an optional SVG overlay
 *  drawn in canvas coordinates (group boxes coloured by inferred role). */
export function Preview({ src, width, height, overlay = [], roleOf, labelOf, showOverlay, className = "", maxHeight }: Props) {
  const [loaded, setLoaded] = useState(false);
  const stroke = Math.max(width, height) / 400;
  const font = Math.max(10, Math.min(width, height) / 22, Math.max(width, height) / 70);
  return (
    <div
      className={`relative overflow-hidden rounded-lg checker shadow-[0_20px_60px_-20px_rgba(0,0,0,.8)] ring-1 ring-white/10 ${className}`}
      style={{ aspectRatio: `${width} / ${height}`, maxHeight, maxWidth: maxHeight ? (maxHeight * width) / height : undefined }}
    >
      {!loaded && <div className="absolute inset-0 skeleton" />}
      {src && (
        <img
          src={src}
          onLoad={() => setLoaded(true)}
          className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-500 ${loaded ? "opacity-100" : "opacity-0"}`}
          alt=""
          draggable={false}
        />
      )}
      {showOverlay && (
        <svg viewBox={`0 0 ${width} ${height}`} className="pointer-events-none absolute inset-0 h-full w-full">
          {overlay.map((b, i) => {
            const role = roleOf?.(b.id);
            if (!role || role === "background") return null;
            const c = roleColor(role);
            const deco = role === "decoration";
            return (
              <g key={`${b.id}-${i}`}>
                <rect
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  fill={deco ? "none" : c}
                  fillOpacity={0.1}
                  stroke={c}
                  strokeWidth={stroke * (deco ? 1 : 2)}
                  strokeDasharray={deco ? `${stroke * 4} ${stroke * 3}` : undefined}
                  rx={stroke * 2}
                />
                {!deco && (
                  <text
                    x={b.x + stroke * 3}
                    y={Math.max(b.y + font, font)}
                    fontSize={font * 0.8}
                    fontFamily="Inter, sans-serif"
                    fontWeight={600}
                    fill={c}
                    style={{ paintOrder: "stroke", stroke: "rgba(0,0,0,.75)", strokeWidth: font * 0.18 }}
                  >
                    {labelOf?.(b.id) ?? role}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
