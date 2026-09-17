import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const DEFAULT_CAMERA = Object.freeze({ position: [0, 20, 125], target: [0, 20, 0] });
const DEFAULT_BODY_CUT = 55;
const params = new URLSearchParams(location.search);
const dataset = params.get('data') || 'changfa';
const manifestPath = params.get('manifest') ||
  'frames_toy_case.json';

class SequenceData {
  async load(onProgress) {
    onProgress('Loading manifest...');
    this.manifest = await fetchJson(manifestPath);

    onProgress('Loading body...');
    const [bodyPositions, bodyIndices] = await Promise.all([
      fetchBuffer(this.manifest.body.pos), fetchBuffer(this.manifest.body.idx),
    ]);

    onProgress('Loading hair...');
    const [hairPositions, hairIndices, hairColors] = await Promise.all([
      fetchBuffer(this.manifest.hair.pos), fetchBuffer(this.manifest.hair.idx),
      fetchBuffer(this.manifest.hair.col),
    ]);

    this.bodyPositions = bodyPositions;
    this.hairPositions = hairPositions;
    this.bodyIndices = new Uint32Array(bodyIndices);
    this.hairIndices = new Uint32Array(hairIndices);
    this.hairColors = new Float32Array(hairColors);
    this.bodyVertexCount = this.manifest.body.vcount;
    this.hairVertexCount = this.manifest.hair.vcount;
    this.strandLengths = this.manifest.hair.strand_lens;
    this.strandIndexOffsets = prefixOffsets(this.strandLengths.map(length => 2 * Math.max(0, length - 1)));
  }

  get frameCount() { return this.manifest.frame_count; }
  get fps() { return this.manifest.fps || 24; }
  get strandCount() { return this.strandLengths.length; }

  strandLength(index) { return this.strandLengths[index]; }

  strandDrawRange(index) {
    return {
      start: this.strandIndexOffsets[index],
      count: 2 * Math.max(0, this.strandLengths[index] - 1),
    };
  }

  pointVertexIndex(strandIndex, pointIndex) {
    const length = this.strandLengths[strandIndex];
    const start = this.strandIndexOffsets[strandIndex];
    if (length <= 1) return 0;
    if (pointIndex < length - 1) return this.hairIndices[start + pointIndex * 2];
    return this.hairIndices[start + (length - 2) * 2 + 1];
  }

  strandFromIndexOffset(indexOffset) {
    let low = 0;
    let high = this.strandLengths.length - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const start = this.strandIndexOffsets[middle];
      const end = start + 2 * Math.max(0, this.strandLengths[middle] - 1);
      if (indexOffset < start) high = middle - 1;
      else if (indexOffset >= end) low = middle + 1;
      else return middle;
    }
    return -1;
  }

  bodyFrame(index) {
    return frameView(this.bodyPositions, index, this.bodyVertexCount);
  }

  hairFrame(index) {
    return frameView(this.hairPositions, index, this.hairVertexCount);
  }
}

class StrandScene {
  constructor(container) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x17191d);
    this.camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 500);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.localClippingEnabled = true;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(innerWidth, innerHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 250;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Line.threshold = 0.6;
    this.resetCamera();
    this.addLights();
    addEventListener('resize', () => this.resize());
  }

  addLights() {
    this.scene.add(new THREE.HemisphereLight(0xdde7ff, 0x34302d, 2.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.5);
    key.position.set(1, 1, 1);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x9eb7e8, 1.5);
    fill.position.set(-1, 0.5, -1);
    this.scene.add(fill);
  }

  build(data) {
    const firstBody = data.bodyFrame(0);
    const firstHair = data.hairFrame(0);

    const bodyGeometry = new THREE.BufferGeometry();
    this.bodyPosition = new THREE.BufferAttribute(new Float32Array(firstBody), 3);
    bodyGeometry.setAttribute('position', this.bodyPosition);
    bodyGeometry.setIndex(new THREE.BufferAttribute(data.bodyIndices, 1));
    bodyGeometry.computeVertexNormals();
    bodyGeometry.computeBoundingBox();
    this.bodyBounds = bodyGeometry.boundingBox.clone();
    this.trackedBodyCenter = bodyCentroid(firstBody);
    this.bodyClipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const bodyMaterial = new THREE.MeshPhongMaterial({
      color: 0xd4a574, specular: 0x171717, shininess: 12,
      side: THREE.DoubleSide, clippingPlanes: [this.bodyClipPlane],
    });
    this.bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
    this.bodyMesh.frustumCulled = false;
    this.scene.add(this.bodyMesh);

    const hairGeometry = new THREE.BufferGeometry();
    this.hairPosition = new THREE.BufferAttribute(new Float32Array(firstHair), 3);
    hairGeometry.setAttribute('position', this.hairPosition);
    hairGeometry.setAttribute('color', new THREE.BufferAttribute(data.hairColors, 3));
    hairGeometry.setIndex(new THREE.BufferAttribute(data.hairIndices, 1));
    hairGeometry.computeBoundingBox();
    this.hairBounds = hairGeometry.boundingBox.clone();
    const hairMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
    this.hairLines = new THREE.LineSegments(hairGeometry, hairMaterial);
    this.hairLines.frustumCulled = false;
    this.scene.add(this.hairLines);

    const pointGeometry = new THREE.BufferGeometry();
    this.selectedPointPosition = new THREE.BufferAttribute(new Float32Array(3), 3);
    pointGeometry.setAttribute('position', this.selectedPointPosition);
    const pointMaterial = new THREE.PointsMaterial({
      color: 0xff3b30, size: 2, sizeAttenuation: true, depthTest: false,
    });
    this.selectedPoint = new THREE.Points(pointGeometry, pointMaterial);
    this.selectedPoint.visible = false;
    this.selectedPoint.renderOrder = 10;
    this.selectedPoint.frustumCulled = false;
    this.scene.add(this.selectedPoint);
    this.resetCamera();
  }

  setFrame(body, hair) {
    this.followBodyTranslation(body);
    this.bodyPosition.array.set(body);
    this.bodyPosition.needsUpdate = true;
    this.bodyMesh.geometry.computeVertexNormals();
    this.hairPosition.array.set(hair);
    this.hairPosition.needsUpdate = true;
  }

  followBodyTranslation(body) {
    const center = bodyCentroid(body);
    if (this.trackedBodyCenter) {
      const offset = center.clone().sub(this.trackedBodyCenter);
      this.camera.position.add(offset);
      this.controls.target.add(offset);
    }
    this.trackedBodyCenter.copy(center);
  }

  setBodyCut(percent) {
    const { min, max } = this.bodyBounds;
    const cutY = THREE.MathUtils.lerp(min.y, max.y, percent / 100);
    this.bodyClipPlane.constant = -cutY;
  }

  showAllHair() {
    this.hairLines.visible = true;
    this.hairLines.geometry.setDrawRange(0, Infinity);
    this.selectedPoint.visible = false;
  }

  showStrand(start, count) {
    this.hairLines.visible = true;
    this.hairLines.geometry.setDrawRange(start, count);
    this.selectedPoint.visible = false;
  }

  showPoint(position) {
    this.hairLines.visible = false;
    this.selectedPointPosition.array.set(position);
    this.selectedPointPosition.needsUpdate = true;
    this.selectedPoint.visible = true;
  }

  pickHair(clientX, clientY) {
    if (!this.hairLines.visible) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    return this.raycaster.intersectObject(this.hairLines, false)[0] || null;
  }

  resetCamera() {
    if (this.bodyBounds && this.hairBounds) {
      const cutY = THREE.MathUtils.lerp(
        this.bodyBounds.min.y, this.bodyBounds.max.y, DEFAULT_BODY_CUT / 100,
      );
      const minX = Math.min(this.bodyBounds.min.x, this.hairBounds.min.x);
      const maxX = Math.max(this.bodyBounds.max.x, this.hairBounds.max.x);
      const maxY = Math.max(this.bodyBounds.max.y, this.hairBounds.max.y);
      const minZ = Math.min(this.bodyBounds.min.z, this.hairBounds.min.z);
      const maxZ = Math.max(this.bodyBounds.max.z, this.hairBounds.max.z);
      const centerX = (minX + maxX) * 0.5;
      const centerY = (cutY + maxY) * 0.5;
      const centerZ = (minZ + maxZ) * 0.5;
      const halfWidth = (maxX - minX) * 0.5;
      const halfHeight = (maxY - cutY) * 0.5;
      const halfFov = THREE.MathUtils.degToRad(this.camera.fov * 0.5);
      const distanceY = halfHeight / Math.tan(halfFov);
      const distanceX = halfWidth / (Math.tan(halfFov) * this.camera.aspect);
      const distance = Math.max(distanceX, distanceY) * 1.08;
      this.controls.target.set(centerX, centerY, centerZ);
      this.camera.position.set(centerX, centerY, maxZ + distance);
      this.controls.update();
      return;
    }
    this.camera.position.fromArray(DEFAULT_CAMERA.position);
    this.controls.target.fromArray(DEFAULT_CAMERA.target);
    this.controls.update();
  }

  resize() {
    this.camera.aspect = innerWidth / innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(innerWidth, innerHeight);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

class ViewerApp {
  constructor() {
    this.ui = bindUi();
    this.scene = new StrandScene(document.querySelector('#viewport'));
    this.data = new SequenceData();
    this.frame = 0;
    this.playing = false;
    this.playbackSpeed = 1;
    this.elapsed = 0;
    this.previousTime = performance.now();
  }

  async start() {
    try {
      await this.data.load(message => this.setStatus(message));
      this.scene.build(this.data);
      this.configureUi();
      this.configureHairSelection();
      this.applyInitialSelection();
      this.setBodyCut(Number(this.ui.bodyCutSlider.value));
      this.setFrame(clampInteger(params.get('frame'), 0, this.data.frameCount - 1));
      this.setStatus(`Loaded ${this.data.frameCount} frame(s)`);
      requestAnimationFrame(time => this.animate(time));
    } catch (error) {
      console.error(error);
      this.setStatus(`Load failed: ${error.message}`);
    }
  }

  configureUi() {
    this.ui.total.textContent = this.data.frameCount;
    this.ui.frameSlider.max = this.data.frameCount - 1;
    this.ui.frameSlider.oninput = event => this.setFrame(Number(event.target.value));
    this.ui.bodyCutSlider.oninput = event => this.setBodyCut(Number(event.target.value));
    this.ui.hairMode.onchange = () => this.updateHairSelection();
    this.ui.strandIndex.oninput = () => this.onStrandChanged();
    this.ui.pointIndex.oninput = () => this.updateHairSelection();
    this.ui.prev.onclick = () => this.setFrame(this.frame - 1);
    this.ui.next.onclick = () => this.setFrame(this.frame + 1);
    this.ui.play.onclick = () => this.togglePlayback();
    this.ui.playbackSpeed.onchange = event => {
      this.playbackSpeed = Number(event.target.value);
      this.elapsed = 0;
    };
    this.ui.body.onclick = () => {
      this.scene.bodyMesh.visible = !this.scene.bodyMesh.visible;
      this.ui.body.textContent = this.scene.bodyMesh.visible ? 'Hide body' : 'Show body';
    };
    this.ui.reset.onclick = () => this.scene.resetCamera();
    this.bindHairPicking();
    addEventListener('keydown', event => this.onKeyDown(event));
  }

  setFrame(index) {
    this.frame = (index + this.data.frameCount) % this.data.frameCount;
    this.scene.setFrame(this.data.bodyFrame(this.frame), this.data.hairFrame(this.frame));
    this.updateHairSelection();
    this.ui.frameSlider.value = this.frame;
    this.ui.frame.textContent = this.frame;
  }

  setBodyCut(percent) {
    this.scene.setBodyCut(percent);
    this.ui.bodyCut.textContent = `${percent}%`;
  }

  configureHairSelection() {
    this.ui.strandIndex.max = this.data.strandCount - 1;
    this.onStrandChanged();
  }

  applyInitialSelection() {
    const mode = params.get('mode');
    if (['all', 'strand', 'point'].includes(mode)) this.ui.hairMode.value = mode;
    if (params.has('strand')) this.ui.strandIndex.value = params.get('strand');
    this.onStrandChanged();
    if (params.has('point')) this.ui.pointIndex.value = params.get('point');
  }

  onStrandChanged() {
    const strand = clampInteger(this.ui.strandIndex.value, 0, this.data.strandCount - 1);
    this.ui.strandIndex.value = strand;
    this.ui.pointIndex.max = this.data.strandLength(strand) - 1;
    this.ui.pointIndex.value = clampInteger(this.ui.pointIndex.value, 0, this.data.strandLength(strand) - 1);
    this.updateHairSelection();
  }

  updateHairSelection() {
    if (!this.scene.hairLines) return;
    const mode = this.ui.hairMode.value;
    const strand = clampInteger(this.ui.strandIndex.value, 0, this.data.strandCount - 1);
    const point = clampInteger(this.ui.pointIndex.value, 0, this.data.strandLength(strand) - 1);
    this.ui.strandIndex.disabled = mode === 'all';
    this.ui.pointIndex.disabled = mode !== 'point';

    if (mode === 'all') {
      this.scene.showAllHair();
    } else if (mode === 'strand') {
      const range = this.data.strandDrawRange(strand);
      this.scene.showStrand(range.start, range.count);
    } else {
      const vertex = this.data.pointVertexIndex(strand, point);
      const positions = this.data.hairFrame(this.frame);
      this.scene.showPoint(positions.subarray(vertex * 3, vertex * 3 + 3));
    }
  }

  bindHairPicking() {
    const canvas = this.scene.renderer.domElement;
    let pointerStart = null;
    canvas.addEventListener('pointerdown', event => {
      pointerStart = { x: event.clientX, y: event.clientY };
    });
    canvas.addEventListener('pointerup', event => {
      if (!pointerStart) return;
      const distance = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y);
      pointerStart = null;
      if (distance > 4) return;
      const hit = this.scene.pickHair(event.clientX, event.clientY);
      if (!hit) return;
      const strand = this.data.strandFromIndexOffset(hit.index);
      if (strand < 0) return;
      this.ui.selectedStrand.textContent = strand;
      this.ui.strandIndex.value = strand;
      this.onStrandChanged();
    });
    canvas.addEventListener('pointercancel', () => { pointerStart = null; });
  }

  togglePlayback() {
    this.playing = !this.playing;
    this.elapsed = 0;
    this.ui.play.textContent = this.playing ? 'Pause' : 'Play';
  }

  onKeyDown(event) {
    if (event.key === 'ArrowLeft') this.setFrame(this.frame - 1);
    if (event.key === 'ArrowRight') this.setFrame(this.frame + 1);
    if (event.key === ' ') {
      event.preventDefault();
      this.togglePlayback();
    }
  }

  animate(time) {
    const delta = Math.min((time - this.previousTime) / 1000, 0.1);
    this.previousTime = time;
    if (this.playing) {
      this.elapsed += delta;
      const frameDuration = 1 / (this.data.fps * this.playbackSpeed);
      while (this.elapsed >= frameDuration) {
        this.setFrame(this.frame + 1);
        this.elapsed -= frameDuration;
      }
    }
    this.scene.render();
    requestAnimationFrame(nextTime => this.animate(nextTime));
  }

  setStatus(message) { this.ui.status.textContent = message; }
}

function bindUi() {
  return {
    frame: document.querySelector('#lblFrame'), total: document.querySelector('#lblTotal'),
    frameSlider: document.querySelector('#frameSlider'), bodyCutSlider: document.querySelector('#bodyCutSlider'),
    hairMode: document.querySelector('#hairMode'), strandIndex: document.querySelector('#strandIndex'),
    pointIndex: document.querySelector('#pointIndex'),
    selectedStrand: document.querySelector('#selectedStrand'),
    bodyCut: document.querySelector('#lblBodyCut'), prev: document.querySelector('#btnPrev'),
    play: document.querySelector('#btnPlay'), next: document.querySelector('#btnNext'),
    playbackSpeed: document.querySelector('#playbackSpeed'),
    body: document.querySelector('#btnBody'), reset: document.querySelector('#btnReset'),
    status: document.querySelector('#status'),
  };
}

function frameView(buffer, frame, vertexCount) {
  return new Float32Array(buffer, frame * vertexCount * 3 * Float32Array.BYTES_PER_ELEMENT, vertexCount * 3);
}

function bodyCentroid(body) {
  let sumX = 0, sumY = 0, sumZ = 0;
  for (let i = 0; i < body.length; i += 3) {
    sumX += body[i];
    sumY += body[i + 1];
    sumZ += body[i + 2];
  }
  const vertexCount = body.length / 3;
  return new THREE.Vector3(sumX / vertexCount, sumY / vertexCount, sumZ / vertexCount);
}

function prefixOffsets(lengths) {
  const offsets = new Uint32Array(lengths.length);
  let offset = 0;
  for (let index = 0; index < lengths.length; index += 1) {
    offsets[index] = offset;
    offset += lengths[index];
  }
  return offsets;
}

function clampInteger(value, min, max) {
  const number = Number.parseInt(value, 10);
  return THREE.MathUtils.clamp(Number.isFinite(number) ? number : min, min, max);
}

async function fetchBuffer(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} (${response.status})`);
  return response.arrayBuffer();
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} (${response.status})`);
  return response.json();
}

new ViewerApp().start();
