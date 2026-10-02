import React from "react";

export function Input({
  label,
  error,
  helperText,
  icon = null,
  type = "text",
  value,
  onChange,
  placeholder = "",
  disabled = false,
  required = false,
  className = "",
  ...props
}) {
  return (
    <div className={`form-group ${className}`}>
      {label && (
        <label className="form-label">
          <span>{label} {required && <span style={{ color: "var(--crimson-primary)" }}>*</span>}</span>
        </label>
      )}
      <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
        {icon && (
          <span
            style={{
              position: "absolute",
              left: "1rem",
              color: "var(--text-faint)",
              display: "flex",
              alignItems: "center",
              pointerEvents: "none",
            }}
          >
            {icon}
          </span>
        )}
        <input
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          className="form-input"
          style={{
            paddingLeft: icon ? "2.75rem" : "1rem",
            borderColor: error ? "var(--crimson-primary)" : undefined,
          }}
          {...props}
        />
      </div>
      {error && (
        <p style={{ color: "var(--crimson-primary)", fontSize: "0.8rem", marginTop: "0.25rem" }}>
          {error}
        </p>
      )}
      {helperText && !error && (
        <p style={{ color: "var(--text-faint)", fontSize: "0.8rem", marginTop: "0.25rem" }}>
          {helperText}
        </p>
      )}
    </div>
  );
}
