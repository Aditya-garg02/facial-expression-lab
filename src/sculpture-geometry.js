import { BufferAttribute, BufferGeometry } from 'three';
import bust from './assets/human-bust.json';

const sampleCount = 30_000;
const dustCount = 460;

function randomSequence() {
  let state = 20261121;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function weightedTriangles(positions, indices) {
  const weights = new Float64Array(indices.length / 3);
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    const a = indices[i * 3] * 3;
    const b = indices[i * 3 + 1] * 3;
    const c = indices[i * 3 + 2] * 3;
    const abx = positions[b] - positions[a];
    const aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a];
    const acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const area = Math.hypot(nx, ny, nz);
    const front = Math.max(0, nz / (area || 1));
    const eye = a / 3 >= bust.eyeStart;
    total += area * (0.46 + 0.92 * front + (eye ? 1.6 : 0));
    weights[i] = total;
  }
  return weights;
}

function pickTriangle(weights, target) {
  let low = 0;
  let high = weights.length - 1;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (weights[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function sampleParticles(surface) {
  const positions = surface.attributes.position.array;
  const normals = surface.attributes.normal.array;
  const indices = surface.index.array;
  const weights = weightedTriangles(positions, indices);
  const total = weights.at(-1);
  const rng = randomSequence();
  const count = sampleCount + dustCount;
  const coordinates = new Float32Array(count * 3);
  const directions = new Float32Array(count * 3);
  const styles = new Float32Array(count * 4);
  const eyes = new Float32Array(count);

  for (let i = 0; i < sampleCount; i++) {
    const triangle = pickTriangle(weights, rng() * total) * 3;
    const a = indices[triangle] * 3;
    const b = indices[triangle + 1] * 3;
    const c = indices[triangle + 2] * 3;
    const root = Math.sqrt(rng());
    const wb = root * (1 - rng());
    const wc = root - wb;
    const wa = 1 - root;
    const offset = i * 3;
    const grain = rng();
    for (let axis = 0; axis < 3; axis++) {
      coordinates[offset + axis] = positions[a + axis] * wa + positions[b + axis] * wb + positions[c + axis] * wc;
      directions[offset + axis] = normals[a + axis] * wa + normals[b + axis] * wb + normals[c + axis] * wc;
    }
    const length = Math.hypot(directions[offset], directions[offset + 1], directions[offset + 2]) || 1;
    const lift = 0.012 + 0.014 * grain;
    for (let axis = 0; axis < 3; axis++) {
      directions[offset + axis] /= length;
      coordinates[offset + axis] += directions[offset + axis] * lift;
    }
    const style = i * 4;
    styles[style] = 0.78 + 1.59 * rng() ** 2.25;
    styles[style + 1] = rng();
    styles[style + 2] = 0.40 + 0.38 * grain;
    styles[style + 3] = grain;
    eyes[i] = a / 3 >= bust.eyeStart ? 1 : 0;
  }

  // A few detached grains make the edges readable when the head turns.
  for (let i = sampleCount; i < count; i++) {
    const angle = rng() * Math.PI * 2;
    const radius = 1.7 + rng() ** 0.55 * 1.32;
    coordinates[i * 3] = Math.cos(angle) * radius;
    coordinates[i * 3 + 1] = Math.sin(angle) * radius * 0.98;
    coordinates[i * 3 + 2] = (rng() - 0.5) * 2.3;
    styles[i * 4] = 0.9 + 1.3 * rng() ** 2;
    styles[i * 4 + 1] = rng();
    styles[i * 4 + 2] = 0.13 + 0.35 * rng() ** 1.4;
    styles[i * 4 + 3] = rng();
  }

  return { positions: coordinates, normals: directions, styles, eyes };
}

export function makeSculpture() {
  const surface = new BufferGeometry();
  const positions = Float32Array.from(bust.positions, (position) => position / 4096);
  const eyes = new Float32Array(positions.length / 3);
  eyes.fill(1, bust.eyeStart);
  surface.setAttribute('position', new BufferAttribute(positions, 3));
  surface.setAttribute('aEye', new BufferAttribute(eyes, 1));
  surface.setIndex(new BufferAttribute(new Uint16Array(bust.indices), 1));
  surface.computeVertexNormals();
  return { surface, points: sampleParticles(surface) };
}
