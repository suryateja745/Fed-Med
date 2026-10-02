import React, { useEffect, useRef } from "react";

/**
 * Animated neural network canvas simulating federated gradient transmissions
 * between decentralized hospital nodes and the central coordinator.
 */
export function NeuralMeshCanvas({ className = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationFrameId;

    const parent = canvas.parentElement;
    let width = (canvas.width = (parent && parent.offsetWidth > 0) ? parent.offsetWidth : window.innerWidth || 800);
    let height = (canvas.height = (parent && parent.offsetHeight > 0) ? parent.offsetHeight : 450);

    const handleResize = () => {
      if (!canvas) return;
      const p = canvas.parentElement;
      width = canvas.width = (p && p.offsetWidth > 0) ? p.offsetWidth : window.innerWidth || 800;
      height = canvas.height = (p && p.offsetHeight > 0) ? p.offsetHeight : 450;
    };
    window.addEventListener("resize", handleResize);

    // Generate node particles
    const particleCount = Math.min(Math.floor(width / 35), 45);
    const particles = [];
    const colors = ["#06B6D4", "#10B981", "#8B5CF6", "#38BDF8"];

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        radius: Math.random() * 2.2 + 1.2,
        color: colors[Math.floor(Math.random() * colors.length)],
        pulse: Math.random() * Math.PI,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 130) {
            const alpha = (1 - dist / 130) * 0.22;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(6, 182, 212, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw and update particles
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.pulse += 0.03;

        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        const currentRadius = p.radius + Math.sin(p.pulse) * 0.5;

        // Outer glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, currentRadius * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = p.color === "#06B6D4" ? "rgba(6, 182, 212, 0.15)" : "rgba(139, 92, 246, 0.12)";
        ctx.fill();

        // Node core
        ctx.beginPath();
        ctx.arc(p.x, p.y, currentRadius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
        opacity: 0.6,
      }}
    />
  );
}
