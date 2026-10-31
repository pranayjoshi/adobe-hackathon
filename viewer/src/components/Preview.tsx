import { useState } from "react";

type Props = {
  src: string | null;
  width: number;
  height: number;
  className?: string;
  maxHeight?: number;
};

/** A rendered design at its true aspect ratio. */
export function Preview({ src, width, height, className = "", maxHeight }: Props) {
  const [loaded, setLoaded] = useState(false);
  return (
    <div
      className={`relative overflow-hidden rounded-lg bg-ink-800 ring-1 ring-white/10 ${className}`}
      style={{ aspectRatio: `${width} / ${height}`, maxHeight, maxWidth: maxHeight ? (maxHeight * width) / height : undefined }}
    >
      {!loaded && <div className="absolute inset-0 skeleton" />}
      {src && (
        <img
          src={src}
          onLoad={() => setLoaded(true)}
          className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
          alt=""
          draggable={false}
        />
      )}
    </div>
  );
}
