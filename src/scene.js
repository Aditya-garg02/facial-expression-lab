import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Group,
  Mesh,
  PerspectiveCamera,
  Plane,
  Points,
  Raycaster,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import { createPepperField } from './pepper-field.js';
import { makeSculpture } from './sculpture-geometry.js';

const fieldOfView = 37;
const viewHeight = 0.06;
const restYaw = -0.36;
const restPitch = -0.025;
const entranceLength = 1040;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const mix = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, t) => {
  const x = clamp((t - a) / (b - a), 0, 1);
  return x * x * (3 - 2 * x);
};
const cameraDistance = (aspect) => 8.9 + Math.max(0, 0.85 - aspect) * 3.45;

const motionShader = `
  uniform float uTime;
  uniform float uEnter;
  uniform vec3 uPulse[2];

  float waveAt(vec3 source, vec2 point) {
    float age = uTime - source.z;
    float trail = distance(source.xy, point) - age * 4.25;
    return exp(-trail * trail * 7.5) * cos(trail * 13.0) * exp(-max(age, 0.0) * 1.18) * step(0.0, age);
  }

  float waveAtSurface(vec3 position) {
    return waveAt(uPulse[0], position.xy) + waveAt(uPulse[1], position.xy);
  }
`;

const lightingShader = `
  float neckline(float height) {
    return smoothstep(-2.91, -1.60, height);
  }

  float irisMask(vec3 p) {
    vec2 center = vec2(0.615, 0.278);
    return 1.0 - smoothstep(0.068, 0.115, distance(vec2(abs(p.x), p.y), center));
  }

  float browPigment(vec3 p) {
    float x = abs(p.x);
    float taper = smoothstep(0.30, 0.43, x) * (1.0 - smoothstep(0.79, 1.0, x));
    float arc = clamp((x - 0.30) / 0.70, 0.0, 1.0);
    float archY = 0.497 + 0.118 * sin(arc * 2.9) - 0.035 * arc + 0.013 * sign(p.x);
    float stroke = (p.y - archY) / (0.049 - 0.019 * arc);
    return taper * exp(-stroke * stroke) * smoothstep(1.15, 1.47, p.z);
  }

  float keyLight(vec3 n) {
    return max(0.0, dot(n, normalize(vec3(-0.72, 0.41, 0.69))));
  }

  float rimLight(vec3 n) {
    return pow(max(0.0, dot(n, normalize(vec3(0.78, 0.08, -0.48)))), 1.7);
  }
`;

const surfaceVertex = `
  attribute float aEye;
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vEye;
  varying float vWave;
  ${motionShader}

  void main() {
    vPosition = position;
    vEye = aEye;
    vWave = waveAtSurface(position);
    vec3 p = position + normal * (0.007 * sin(uTime * 0.92 + position.y) + vWave * 0.105);
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const surfaceFragment = `
  uniform float uEnter;
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vEye;
  varying float vWave;
  ${lightingShader}

  void main() {
    vec3 n = normalize(vNormal);
    float grain = fract(sin(dot(gl_FragCoord.xy, vec2(13.42, 77.19))) * 43758.5453);
    float key = pow(keyLight(n), 1.21);
    float fill = max(0.0, dot(n, normalize(vec3(0.78, -0.2, 0.58))));
    float rim = rimLight(n);
    vec3 clay = vec3(0.103, 0.065, 0.060) + key * vec3(0.43, 0.29, 0.205)
      + fill * vec3(0.075, 0.046, 0.062) + rim * vec3(0.215, 0.080, 0.053);
    vec3 eye = mix(vec3(0.332, 0.255, 0.212), vec3(0.153, 0.100, 0.076), irisMask(vPosition));
    eye *= (0.50 + 0.55 * key + 0.20 * fill);
    clay *= 1.0 - browPigment(vPosition) * (0.36 + grain * 0.10);
    vec3 tone = mix(clay, eye, vEye);
    tone += (grain - 0.5) * 0.085 + abs(vWave) * vec3(0.1, 0.043, 0.018);
    float fade = neckline(vPosition.y) * (1.0 - smoothstep(0.1, 0.48, uEnter));
    if (fade < 0.005) discard;
    gl_FragColor = vec4(tone, fade);
  }
`;

const particleVertex = `
  attribute vec4 aStyle;
  attribute float aEye;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uFlash;
  varying float vAlpha;
  varying vec3 vColor;
  ${motionShader}
  ${lightingShader}

  void main() {
    float grain = aStyle.w;
    float dust = 1.0 - step(0.1, length(normal));
    float wave = waveAtSurface(position);
    float pulse = min(abs(wave), 1.0);
    float movement = 0.007 * sin(uTime * 0.92 + position.y) + wave * (0.112 + 0.094 * grain);
    vec3 p = position + normal * movement;
    p.y += dust * 0.10 * sin(uTime * 0.68 + grain * 42.0);

    vec4 view = modelViewMatrix * vec4(p, 1.0);
    vec3 n = normalize(normalMatrix * normal + vec3(0.0, 0.0, 0.0001));
    float key = keyLight(n);
    float rim = rimLight(n);
    float facing = max(0.0, dot(n, normalize(-view.xyz)));
    float brightness = (0.25 + 0.86 * key + rim * 0.64) * (0.27 + 0.73 * smoothstep(0.0, 0.55, facing));
    brightness = mix(brightness, 0.55, dust);
    brightness *= mix(1.0, mix(0.76, 0.34, irisMask(position)), aEye);
    brightness *= 1.0 - browPigment(position) * 0.62;

    vec3 copper = vec3(0.91, 0.35, 0.21);
    vec3 bone = vec3(1.0, 0.84, 0.64);
    vec3 glint = vec3(0.76, 0.70, 0.85);
    vColor = mix(copper, bone, smoothstep(0.08, 0.88, aStyle.y));
    vColor = mix(vColor, glint, smoothstep(0.93, 1.0, aStyle.y) * 0.70);
    vColor = mix(vColor, vec3(1.0, 0.86, 0.72), min(0.85, pulse * 0.56 + uFlash * 0.44));

    vec4 projected = projectionMatrix * view;
    float burst = smoothstep(0.13 + grain * 0.10, 0.74 + grain * 0.08, uEnter);
    float angle = grain * 125.6637 + atan(p.y, p.x) * 0.62;
    float spread = 0.44 + 1.16 * fract(aStyle.y * 7.19 + grain);
    vec2 exitPoint = vec2(cos(angle) * 1.18, sin(angle) * 1.28) * spread;
    float clipW = max(0.25, projected.w);
    vec2 screen = clamp(projected.xy / clipW, vec2(-2.0), vec2(2.0));
    projected.w = mix(projected.w, 1.0, burst);
    projected.xy = mix(screen, exitPoint, burst) * projected.w;
    projected.z = mix(projected.z, 0.0, burst);

    float strength = aStyle.z * brightness * mix(neckline(position.y), 1.0, dust);
    strength += pulse * 0.25 * max(facing, dust);
    strength = mix(strength, aStyle.z * (0.33 + 0.68 * grain), burst * 0.84);
    vAlpha = strength * (1.0 - smoothstep(0.79, 1.0, uEnter));

    float perspective = mix(clamp(8.9 / max(1.0, -view.z), 0.64, 3.8), 1.7 + grain * 0.95, burst);
    gl_PointSize = uPixelRatio * uPointScale * (aStyle.x + pulse * 0.92 + burst * grain * 0.70) * perspective;
    gl_Position = projected;
  }
`;

const particleFragment = `
  varying float vAlpha;
  varying vec3 vColor;

  void main() {
    float distanceFromCenter = length(gl_PointCoord - 0.5) * 2.0;
    float grain = 1.0 - smoothstep(0.36, 1.0, distanceFromCenter);
    gl_FragColor = vec4(vColor, grain * vAlpha);
  }
`;

function makeCanvas() {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { display: 'block', width: '100%', height: '100%' });
  return canvas;
}

function createWebGL(canvas, sculpture) {
  const context = canvas.getContext('webgl2', { alpha: true });
  if (!context) return null;

  const renderer = new WebGLRenderer({ canvas, context, alpha: true });
  const camera = new PerspectiveCamera(fieldOfView, 1, 0.16, 32);
  const scene = new Scene();
  const portrait = new Group();
  scene.add(portrait);

  const particles = new BufferGeometry();
  particles.setAttribute('position', new BufferAttribute(sculpture.points.positions, 3));
  particles.setAttribute('normal', new BufferAttribute(sculpture.points.normals, 3));
  particles.setAttribute('aStyle', new BufferAttribute(sculpture.points.styles, 4));
  particles.setAttribute('aEye', new BufferAttribute(sculpture.points.eyes, 1));

  const motion = {
    uTime: { value: 0 },
    uEnter: { value: 0 },
    uPulse: { value: [new Vector3(0, 0, -30), new Vector3(0, 0, -30)] },
  };
  const surfaceMaterial = new ShaderMaterial({
    vertexShader: surfaceVertex,
    fragmentShader: surfaceFragment,
    uniforms: motion,
    transparent: true,
  });
  const pointMaterial = new ShaderMaterial({
    vertexShader: particleVertex,
    fragmentShader: particleFragment,
    uniforms: {
      ...motion,
      uPixelRatio: { value: 1 },
      uPointScale: { value: 1 },
      uFlash: { value: 0 },
    },
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
  });

  const surface = new Mesh(sculpture.surface, surfaceMaterial);
  const field = new Points(particles, pointMaterial);
  field.frustumCulled = false;
  field.renderOrder = 1;
  portrait.add(surface, field);

  const ray = new Raycaster();
  const plane = new Plane(new Vector3(0, 0, 1), -1.05);
  const hit = new Vector3();
  const pointer = new Vector2();
  let width = 1;
  let height = 1;

  return {
    animated: true,
    resize(nextWidth, nextHeight, ratio) {
      width = nextWidth;
      height = nextHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      pointMaterial.uniforms.uPixelRatio.value = ratio;
      pointMaterial.uniforms.uPointScale.value = clamp(height / 590, 0.70, 1.17);
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
    },
    render({ time, yaw, pitch, entrance, origin, flash }) {
      let distance = cameraDistance(width / height);
      if (origin) {
        // Keep the same projection when the host is promoted to the full viewport.
        const spread = smoothstep(0.04, 0.73, entrance);
        const frameWidth = mix(origin.width, width, spread);
        const frameHeight = mix(origin.height, height, spread);
        camera.setViewOffset(frameWidth, frameHeight, -origin.left * (1 - spread), -origin.top * (1 - spread), width, height);
        distance = cameraDistance(origin.width / origin.height);
      }
      camera.position.set(0, viewHeight, mix(distance, 1.68, entrance ** 1.81));
      camera.lookAt(0, viewHeight, 0);
      portrait.rotation.set(pitch - entrance * 0.08, yaw - entrance * 0.34, entrance * -0.055);
      motion.uTime.value = time;
      motion.uEnter.value = entrance;
      pointMaterial.uniforms.uFlash.value = flash;
      pointMaterial.depthTest = entrance < 0.24;
      surface.visible = entrance < 0.48;
      renderer.render(scene, camera);
    },
    pulse(x, y, time) {
      ray.setFromCamera(pointer.set(x, y), camera);
      if (!ray.ray.intersectPlane(plane, hit)) return;
      portrait.worldToLocal(hit);
      const [latest, previous] = motion.uPulse.value;
      previous.copy(latest);
      latest.set(clamp(hit.x, -2.8, 2.8), clamp(hit.y, -3, 3), time);
    },
    destroy() {
      particles.dispose();
      surfaceMaterial.dispose();
      pointMaterial.dispose();
      renderer.dispose();
      if (!context.isContextLost()) renderer.forceContextLoss();
    },
  };
}

function createStatic(canvas, points) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  let width = 1;
  let height = 1;
  let ratio = 1;
  let pulse = [0, 0];
  const sin = Math.sin(restYaw);
  const cos = Math.cos(restYaw);

  return {
    animated: false,
    resize(nextWidth, nextHeight, nextRatio) {
      width = nextWidth;
      height = nextHeight;
      ratio = nextRatio;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    },
    render({ flash }) {
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.globalCompositeOperation = 'lighter';
      const focal = height / (2 * Math.tan(fieldOfView * Math.PI / 360));
      const depth = cameraDistance(width / height);
      for (let i = 0; i < points.positions.length / 3; i++) {
        const x = points.positions[i * 3];
        const y = points.positions[i * 3 + 1];
        const z = points.positions[i * 3 + 2];
        const nx = points.normals[i * 3];
        const ny = points.normals[i * 3 + 1];
        const nz = points.normals[i * 3 + 2];
        const normalX = nx * cos + nz * sin;
        const normalZ = nz * cos - nx * sin;
        const dust = !nx && !ny && !nz;
        const key = clamp(normalZ * 0.69 + ny * 0.41 - normalX * 0.72, 0, 1);
        const brightness = dust ? 0.59 : clamp((normalZ + 0.05) / 0.54, 0, 1) * (0.44 + 0.97 * key);
        const eye = points.eyes[i] && Math.hypot(Math.abs(x) - 0.615, y - 0.278) < 0.093;
        const alpha = points.styles[i * 4 + 2] * brightness * (dust ? 1 : smoothstep(-2.91, -1.6, y)) * (eye ? 0.2 : 1);
        if (alpha < 0.04) continue;

        const distance = depth - (z * cos - x * sin);
        const screenX = width / 2 + (x * cos + z * sin) * focal / distance;
        const screenY = height / 2 - (y - viewHeight) * focal / distance;
        const size = points.styles[i * 4] * 8.9 / distance * clamp(height / 590, 0.7, 1.17);
        const tint = points.styles[i * 4 + 1];
        const color = tint > 0.94 ? '193, 176, 214' : `255, ${Math.round(105 + 100 * tint)}, ${Math.round(61 + 96 * tint)}`;
        const glow = flash * Math.exp(-Math.abs(Math.hypot(screenX - pulse[0], screenY - pulse[1]) - width * 0.14) * 0.09);
        ctx.fillStyle = `rgba(${color}, ${Math.min(1, alpha * 0.72 + glow * 0.35)})`;
        ctx.fillRect(screenX - size / 2, screenY - size / 2, size, size);
      }
    },
    pulse(x, y) {
      pulse = [(x + 1) * width / 2, (1 - y) * height / 2];
    },
    destroy() {},
  };
}

export function createSculpture(host, pepperHost) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const listeners = new AbortController();
  const { signal } = listeners;
  const sculpture = makeSculpture();
  const timeOrigin = performance.now();
  let canvas = makeCanvas();
  let backend;
  let visible = true;
  let frame = 0;
  let resetFlash = 0;
  let lastFrame = 0;
  let entranceAt = null;
  let entranceOrigin = null;
  let targetX = 0;
  let targetY = 0;
  let yaw = restYaw;
  let pitch = restPitch;
  let flash = 0;

  function seconds(now) {
    return (now - timeOrigin) / 1000;
  }

  function useStaticCanvas() {
    backend?.destroy();
    const previous = canvas;
    canvas = makeCanvas();
    previous.replaceWith(canvas);
    backend = createStatic(canvas, sculpture.points);
    resize();
  }

  canvas.addEventListener('webglcontextlost', useStaticCanvas, { signal });
  host.append(canvas);
  try {
    backend = createWebGL(canvas, sculpture);
  } catch {
    backend = null;
  }
  if (!backend) useStaticCanvas();
  const pepper = createPepperField(pepperHost);

  function render(now) {
    frame = 0;
    if (!backend || (!visible && entranceAt === null) || document.hidden) return;
    const elapsed = Math.min(0.08, (now - lastFrame) / 1000);
    const time = seconds(now);
    const motion = backend.animated && !reducedMotion.matches;
    const entrance = entranceAt === null ? 0 : clamp((now - entranceAt) / entranceLength, 0, 1);
    lastFrame = now;

    if (motion) {
      const ease = 1 - Math.exp(-elapsed * 5.2);
      yaw = mix(yaw, restYaw + targetX * 0.25 + Math.sin(time * 0.70) * 0.031, ease);
      pitch = mix(pitch, restPitch - targetY * 0.16 + Math.sin(time * 0.52) * 0.019, ease);
    } else {
      yaw = restYaw;
      pitch = restPitch;
    }

    backend.render({ time: motion ? time : 0, yaw, pitch, entrance: motion ? entrance : 0, origin: entranceOrigin, flash });
    if (motion && entrance < 1) requestRender();
  }

  function requestRender() {
    if (!frame && backend && (visible || entranceAt !== null) && !document.hidden) frame = requestAnimationFrame(render);
  }

  function resize() {
    const { clientWidth: width, clientHeight: height } = host;
    if (!backend || !width || !height) return;
    const renderBudget = Math.sqrt(3_250_000 / (width * height));
    backend.resize(width, height, Math.min(devicePixelRatio || 1, width < 540 ? 1.45 : 1.75, renderBudget));
    requestRender();
  }

  function pulse(x = 0, y = 0) {
    if (!backend || entranceAt !== null) return;
    backend.pulse(x, y, seconds(performance.now()));
    if (!backend.animated || reducedMotion.matches) {
      flash = 1;
      clearTimeout(resetFlash);
      resetFlash = window.setTimeout(() => {
        flash = 0;
        requestRender();
      }, 180);
    }
    requestRender();
  }

  function localPointer(event, region = host) {
    const rect = region.getBoundingClientRect();
    if (!rect.width || !rect.height) return [0, 0];
    return [(event.clientX - rect.left) / rect.width * 2 - 1, 1 - (event.clientY - rect.top) / rect.height * 2];
  }

  function resetTarget() {
    targetX = 0;
    targetY = 0;
  }

  const pointerTarget = pepper ? window : host;
  pointerTarget.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch' || !event.isPrimary) return;
    const [x, y] = localPointer(event, pepper ? pepperHost : host);
    if (Math.abs(x) <= 1 && Math.abs(y) <= 1) {
      targetX = x;
      targetY = y;
    } else {
      resetTarget();
    }
  }, { signal, passive: true });
  pointerTarget.addEventListener(pepper ? 'pointerout' : 'pointerleave', (event) => {
    if (!pepper || !event.relatedTarget) resetTarget();
  }, { signal });
  host.addEventListener('pointerdown', (event) => {
    if (event.isPrimary && event.button === 0) pulse(...localPointer(event));
  }, { signal, passive: true });

  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    requestRender();
  });
  observer.observe(host);
  const sizing = new ResizeObserver(resize);
  sizing.observe(host);
  reducedMotion.addEventListener('change', requestRender, { signal });
  document.addEventListener('visibilitychange', requestRender, { signal });
  resize();

  return {
    get animated() { return Boolean(backend?.animated); },
    enterDuration: entranceLength,
    pulse: backend || pepper ? () => {
      const rect = host.getBoundingClientRect();
      pepper?.pulse(rect.left + rect.width / 2, rect.top + rect.height / 2);
      pulse();
    } : undefined,
    enter(origin) {
      if (entranceAt !== null) return;
      entranceAt = performance.now();
      entranceOrigin = origin?.width && origin.height ? origin : null;
      resize();
    },
    destroy() {
      listeners.abort();
      observer.disconnect();
      sizing.disconnect();
      cancelAnimationFrame(frame);
      clearTimeout(resetFlash);
      pepper?.destroy();
      backend?.destroy();
      sculpture.surface.dispose();
      backend = null;
      canvas.remove();
    },
  };
}
