import React from "react";

export function Card({
  title,
  subtitle,
  headerAction,
  icon,
  glow = null,
  className = "",
  children,
  ...props
}) {
  const glowClass = glow === "cyan" ? "card-glow-cyan" : glow === "emerald" ? "card-glow-emerald" : "";

  return (
    <div className={`card ${glowClass} ${className}`} {...props}>
      {(title || headerAction) && (
        <div className="card-header">
          <div>
            <div className="card-title">
              {icon && <span style={{ color: "var(--cyan-primary)" }}>{icon}</span>}
              <span>{title}</span>
            </div>
            {subtitle && (
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  );
}
