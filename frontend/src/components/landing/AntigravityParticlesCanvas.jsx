import React, { useEffect, useRef } from "react";

export function AntigravityParticlesCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId;
    let width = (canvas.width = canvas.parentElement.offsetWidth);
    let height = (canvas.height = canvas.parentElement.offsetHeight);

    const colors = [
      "#4285F4", // Google Blue
      "#2563EB", // Royal Blue
      "#3B82F6", // Sky Cerulean
      "#6366F1", // Indigo
      "#8B5CF6", // Violet
      "#A855F7", // Purple
      "#EC4899", // Vibrant Pink
      "#F43F5E", // Rose
      "#06B6D4", // Cyan
      "#10B981", // Emerald
      "#94A3B8", // Soft Slate
    ];

    // Mouse tracking with smooth lerp
    const mouse = {
      x: -1000,
      y: -1000,
      targetX: -1000,
      targetY: -1000,
      radius: 200,
      active: false,
    };

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.targetX = e.clientX - rect.left;
      mouse.targetY = e.clientY - rect.top;
      mouse.active = true;
    };

    const handleMouseLeave = () => {
      mouse.targetX = -1000;
      mouse.targetY = -1000;
      mouse.active = false;
    };

    window.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.offsetWidth;
      height = canvas.height = canvas.parentElement.offsetHeight;
      initParticles();
    };

    window.addEventListener("resize", handleResize);

    // Particle class simulating the Antigravity radial field
    class Particle {
      constructor(radius, angle, color) {
        this.radius = radius;
        this.angle = angle;
        this.color = color;
        this.speed = (0.0003 + Math.random() * 0.0006) * (Math.random() > 0.5 ? 1 : -1);
        this.dashLength = 4 + Math.random() * 7;
        this.dashWidth = 2 + Math.random() * 1.5;

        // Current and anchor coordinates
        const centerX = width / 2;
        const centerY = height * 0.46;
        this.x = centerX + Math.cos(this.angle) * this.radius;
        this.y = centerY + Math.sin(this.angle) * (this.radius * 0.65); // Elliptical perspective
        this.originX = this.x;
        this.originY = this.y;

        this.vx = 0;
        this.vy = 0;
        this.friction = 0.88;
        this.spring = 0.04;
      }

      update(centerX, centerY) {
        // Continuous slow orbit rotation
        this.angle += this.speed;
        this.originX = centerX + Math.cos(this.angle) * this.radius;
        this.originY = centerY + Math.sin(this.angle) * (this.radius * 0.65);

        // Interaction with mouse cursor (repulsion + gentle swirl)
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < mouse.radius && mouse.active) {
          const force = (1 - dist / mouse.radius) * 14;
          const angle = Math.atan2(dy, dx);
          // Radial push
          this.vx += Math.cos(angle) * force;
          this.vy += Math.sin(angle) * force;
          // Subtle tangential swirl
          this.vx += -Math.sin(angle) * (force * 0.4);
          this.vy += Math.cos(angle) * (force * 0.4);
        }

        // Spring force towards anchor origin
        const homeDx = this.originX - this.x;
        const homeDy = this.originY - this.y;
        this.vx += homeDx * this.spring;
        this.vy += homeDy * this.spring;

        // Apply friction
        this.vx *= this.friction;
        this.vy *= this.friction;

        // Update positions
        this.x += this.vx;
        this.y += this.vy;
      }

      draw() {
        ctx.save();
        ctx.translate(this.x, this.y);

        // Orient dash along radial dispersion angle
        const centerX = width / 2;
        const centerY = height * 0.46;
        const tangentAngle = Math.atan2(this.y - centerY, this.x - centerX);
        ctx.rotate(tangentAngle);

        // Draw smooth rounded capsule / dash
        ctx.beginPath();
        const r = this.dashWidth / 2;
        const l = this.dashLength;
        ctx.fillStyle = this.color;
        ctx.roundRect(-l / 2, -r, l, this.dashWidth, r);
        ctx.fill();

        ctx.restore();
      }
    }

    let particles = [];
    const initParticles = () => {
      particles = [];
      const particleCount = Math.min(380, Math.floor((width * height) / 2200));
      const minRadius = 120;
      const maxRadius = Math.max(width, height) * 0.72;

      for (let i = 0; i < particleCount; i++) {
        // Exponential distribution for higher density towards center and smooth outward dispersal
        const rNorm = Math.pow(Math.random(), 1.25);
        const radius = minRadius + rNorm * (maxRadius - minRadius);
        const angle = Math.random() * Math.PI * 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        particles.push(new Particle(radius, angle, color));
      }
    };

    initParticles();

    // Render loop
    const render = () => {
      // Smooth mouse interpolation
      mouse.x += (mouse.targetX - mouse.x) * 0.15;
      mouse.y += (mouse.targetY - mouse.y) * 0.15;

      ctx.clearRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height * 0.46;

      for (let i = 0; i < particles.length; i++) {
        particles[i].update(centerX, centerY);
        particles[i].draw();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
      }}
    />
  );
}
