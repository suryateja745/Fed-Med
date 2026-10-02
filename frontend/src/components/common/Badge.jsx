import React from "react";

export function Badge({
  children,
  variant = "cyan",
  pulse = false,
  icon = null,
  className = "",
  ...props
}) {
  const variantClass = `badge-${variant}`;
  const pulseClass = variant === "emerald" ? "pulse-dot-emerald" : variant === "crimson" ? "pulse-dot-crimson" : "pulse-dot-cyan";

  return (
    <span className={`badge ${variantClass} ${className}`} {...props}>
      {pulse && <span className={`pulse-dot ${pulseClass}`} />}
      {icon && <span style={{ display: "inline-flex", alignItems: "center" }}>{icon}</span>}
      <span>{children}</span>
    </span>
  );
}
