const particles = [
  { radius: 0.54, color: '#bc8a7373' },
  { radius: 0.79, color: '#d392767a' },
  { radius: 1.07, color: '#e6ac9090' },
  { radius: 1.42, color: '#f5c3a5b0' },
];

const maxPulses = 4;
const pulseLifetime = 2050;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const ease = (value) => value * value * (3 - 2 * value);

function seededRandom() {
  let seed = 29_741;
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function createGrains(width, height) {
  const random = seededRandom();
  const count = clamp(Math.round(width * height / 1460), 345, 1880);
  const columns = Math.ceil(Math.sqrt(count * width / height));
  const rows = Math.ceil(count / columns);
  const groups = particles.map(() => []);

  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const tone = random();
      const group = tone < 0.54 ? 0 : tone < 0.86 ? 1 : tone < 0.97 ? 2 : 3;
      groups[group].push({
        x: (column + random()) * width / columns,
        y: (row + random()) * height / rows,
        seed: random(),
      });
    }
  }
  return groups;
}

function drawFront(context, pulse) {
  if (pulse.radius < 9) return;
  context.beginPath();
  const variation = Math.min(8, pulse.radius * 0.035);
  for (let index = 0; index <= 60; index++) {
    const theta = index * Math.PI / 30;
    const edge = pulse.radius + variation * Math.sin(theta * 4 + pulse.phase) * Math.sin(theta * 2 - pulse.phase);
    const x = pulse.x + Math.cos(theta) * edge;
    const y = pulse.y + Math.sin(theta) * edge;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.closePath();
  context.globalAlpha = pulse.energy * 0.15;
  context.lineWidth = 0.9;
  context.strokeStyle = '#ffb794';
  context.stroke();
  context.globalAlpha = 1;
}

export function createPepperField(host) {
  if (!host) return null;

  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  const context = canvas.getContext('2d', { alpha: true });
  if (!context) return null;

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const listeners = new AbortController();
  const { signal } = listeners;
  const pulses = [];
  const pointer = { x: 0, y: 0, targetX: 0, targetY: 0, presence: 0, inside: false };
  let drag = null;
  let groups = [];
  let width = 0;
  let height = 0;
  let dpr = 1;
  let diagonal = 1;
  let visible = true;
  let frame = 0;
  let lastFrame = 0;

  host.append(canvas);

  function draw(now) {
    frame = 0;
    if (!visible || document.hidden || !width || !height) return;
    const moving = !motion.matches;
    const interval = pulses.length || pointer.presence > 0.05 ? 0 : 33;
    if (moving && now - lastFrame < interval) {
      schedule();
      return;
    }
    const dt = Math.min(80, now - lastFrame);
    lastFrame = now;
    const seconds = moving ? now / 1000 : 0;
    const follow = 1 - Math.exp(-dt / 47);
    pointer.presence += ((pointer.inside ? 1 : 0) - pointer.presence) * follow;
    pointer.x += (pointer.targetX - pointer.x) * follow;
    pointer.y += (pointer.targetY - pointer.y) * follow;
    const mouseRadius = clamp(width * 0.13, 94, 174);

    while (pulses.length && now - pulses[0].started > pulseLifetime) pulses.shift();
    for (const pulse of pulses) {
      const elapsed = clamp((now - pulse.started) / pulseLifetime, 0, 1);
      pulse.radius = 8 + diagonal * 1.6 * (1 - Math.exp(-elapsed * 2.5));
      pulse.energy = Math.pow(1 - elapsed, 1.55) * (1 - Math.exp(-elapsed * 48)) * pulse.pressure;
      pulse.phase = elapsed * 5;
    }

    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    for (const pulse of pulses) drawFront(context, pulse);

    for (let group = 0; group < groups.length; group++) {
      const style = particles[group];
      context.beginPath();
      for (const grain of groups[group]) {
        let x = grain.x;
        let y = grain.y;
        let sheen = 0;
        if (moving) {
          const phase = grain.seed * 6.283;
          x += Math.sin(seconds * 0.54 + grain.y * 0.012 + phase) * 3.6;
          y += Math.cos(seconds * 0.43 + grain.x * 0.009 + phase) * 3.2;

          if (pointer.presence > 0.01) {
            const dx = x - pointer.x;
            const dy = y - pointer.y;
            if (Math.abs(dx) < mouseRadius && Math.abs(dy) < mouseRadius) {
              const distance = Math.hypot(dx, dy);
              const influence = ease(clamp(1 - distance / mouseRadius, 0, 1)) * pointer.presence;
              const scale = influence * 55 / Math.max(distance, 1);
              x += (dx - dy * 0.42) * scale;
              y += (dy + dx * 0.42) * scale;
              sheen += influence * 0.5;
            }
          }

          for (const pulse of pulses) {
            const dx = grain.x - pulse.x;
            const dy = grain.y - pulse.y;
            const distance = Math.hypot(dx, dy);
            const depth = pulse.radius - distance;
            if (depth < -56) continue;

            const edge = Math.exp(-Math.pow(depth / 45, 2));
            const crossed = ease(clamp((depth + 24) / 58, 0, 1));
            const force = (crossed * 1.2 + edge * 0.78) * pulse.energy * pulse.reach
              * Math.exp(-distance / (diagonal * 0.4)) * (0.72 + grain.seed * 0.52);
            const radial = force / Math.max(distance, 1);
            const eddy = radial * edge * Math.sin(phase + pulse.phase * 2) * 0.41;
            x += dx * radial - dy * eddy;
            y += dy * radial + dx * eddy;
            sheen += edge * pulse.energy * 0.8;
          }
        }

        if (x < -3 || x > width + 3 || y < -3 || y > height + 3) continue;
        const radius = style.radius * (1 + Math.min(0.9, sheen));
        context.moveTo(x + radius, y);
        context.arc(x, y, radius, 0, 2 * Math.PI);
      }
      context.fillStyle = style.color;
      context.fill();
    }
    if (moving) schedule();
  }

  function schedule() {
    if (!frame && visible && !document.hidden) frame = requestAnimationFrame(draw);
  }

  function resize() {
    const { clientWidth, clientHeight } = host;
    if (!clientWidth || !clientHeight) return;
    width = clientWidth;
    height = clientHeight;
    diagonal = Math.hypot(width, height);
    dpr = Math.min(devicePixelRatio || 1, 1.5, Math.sqrt(2_200_000 / (width * height)));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    groups = createGrains(width, height);
    schedule();
  }

  function localPoint(clientX, clientY) {
    const rect = host.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    return x >= 0 && y >= 0 && x <= rect.width && y <= rect.height ? { x, y } : null;
  }

  function addPulse(point, pressure) {
    if (!point || motion.matches) return;
    if (pulses.length === maxPulses) pulses.shift();
    pulses.push({
      ...point,
      started: performance.now(),
      pressure,
      reach: clamp(diagonal * 0.079, 61, 153),
      radius: 0,
      energy: 0,
      phase: 0,
    });
    schedule();
  }

  function movePointer(event) {
    if (motion.matches || !visible || !event.isPrimary) return;
    const point = localPoint(event.clientX, event.clientY);
    pointer.inside = Boolean(point) && event.pointerType !== 'touch';
    if (!point) return;

    if (pointer.inside) {
      if (pointer.presence < 0.05) {
        pointer.x = point.x;
        pointer.y = point.y;
      }
      pointer.targetX = point.x;
      pointer.targetY = point.y;
    }
    if (drag && performance.now() - drag.at > 105 && Math.hypot(drag.x - point.x, drag.y - point.y) > 35) {
      addPulse(point, 0.52);
      drag = { ...point, at: performance.now() };
    }
    schedule();
  }

  function startPointer(event) {
    if (motion.matches || !visible || !event.isPrimary || event.button !== 0) return;
    if (event.target.closest?.('a, button, input, select, textarea, label')) return;
    const point = localPoint(event.clientX, event.clientY);
    if (!point) return;
    addPulse(point, 1);
    drag = { ...point, at: performance.now() };
  }

  function stopPointer(event) {
    if (event.isPrimary) drag = null;
  }

  const options = { signal, passive: true };
  window.addEventListener('pointermove', movePointer, options);
  window.addEventListener('pointerdown', startPointer, options);
  window.addEventListener('pointerup', stopPointer, options);
  window.addEventListener('pointercancel', stopPointer, options);
  window.addEventListener('pointerout', (event) => {
    if (event.relatedTarget) return;
    pointer.inside = false;
    drag = null;
  }, options);
  document.addEventListener('visibilitychange', schedule, { signal });
  motion.addEventListener('change', () => {
    pulses.length = 0;
    pointer.inside = false;
    pointer.presence = 0;
    drag = null;
    schedule();
  }, { signal });
  const intersection = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) pointer.inside = false;
    schedule();
  });
  const sizing = new ResizeObserver(resize);
  intersection.observe(host);
  sizing.observe(host);
  resize();

  return {
    pulse(clientX, clientY) {
      addPulse(localPoint(clientX, clientY) ?? { x: width / 2, y: height / 2 }, 1);
    },
    destroy() {
      listeners.abort();
      intersection.disconnect();
      sizing.disconnect();
      cancelAnimationFrame(frame);
      canvas.remove();
    },
  };
}
