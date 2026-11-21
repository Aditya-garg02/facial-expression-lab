const $ = selector => document.querySelector(selector);
const video = $('video'), view = $('canvas'), ctx = view.getContext('2d');
const status = $('#status'), camera = $('#camera'), result = $('#result');
const capture = $('#capture'), retake = $('#retake');
const labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
function surface(size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  return canvas.getContext('2d', {willReadFrequently: true});
}
const detectorInput = surface(640), crop = surface(112);
let runtime, stream, generation = 0;
function models() {
  if (!runtime) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = typeof Worker === 'function';
    ort.env.wasm.wasmPaths = new URL('./', location.href).href;
    runtime = Promise.all([
      ort.InferenceSession.create('detector.onnx'),
      ort.InferenceSession.create('mobilefacenet.onnx'),

    ]).catch(error => { runtime = undefined; throw error; });
  }
  return runtime;
}
function reset() {
  generation++;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  camera.hidden = false;
  video.hidden = true;
  camera.disabled = false;
  view.classList.remove('selfie');
  camera.textContent = 'Start camera';
  capture.hidden = retake.hidden = true;
  result.hidden = view.hidden = true;
  return generation;
}
function selectFace(outputs, scale, w, h) {
  let best;
  let bestScore = 0.8;
  for (const stride of [8, 16, 32]) {
    const classes = outputs['cls_' + stride].data;
    const objects = outputs['obj_' + stride].data;
    const boxes = outputs['bbox_' + stride].data;
    const columns = 640 / stride;
    for (let i = 0; i < classes.length; i++) {
      const confidence = Math.sqrt(classes[i] * objects[i]);
      if (!(confidence > bestScore)) continue;
      // YuNet encodes center offsets in grid cells and dimensions in log space.
      const k = i * 4;
      const unit = stride / scale;
      const cx = (i % columns + boxes[k]) * unit;
      const cy = (Math.floor(i / columns) + boxes[k + 1]) * unit;
      const width = Math.exp(boxes[k + 2]) * unit;
      const height = Math.exp(boxes[k + 3]) * unit;
      const x = Math.max(0, cx - width / 2);
      const y = Math.max(0, cy - height / 2);
      const boxWidth = Math.min(w, cx + width / 2) - x;
      const boxHeight = Math.min(h, cy + height / 2) - y;
      if (Math.min(boxWidth, boxHeight) < 24) continue;
      bestScore = confidence;
      const offsets = outputs['kps_' + stride].data;
      const points = [];
      for (let j = 0; j < 5; j++) {
        const offset = i * 10 + j * 2;
        points.push([(i % columns + offsets[offset]) * unit,
          (Math.floor(i / columns) + offsets[offset+1]) * unit]);
      }
      best = {box: [x,y,boxWidth,boxHeight], points};
    }
  }
  return best;
}
// Eye, nose and mouth landmarks in the model's 112px reference crop.
const reference = [[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
  [41.5493,92.3655],[70.7299,92.2041]];
function centroid(points) {
  return points.reduce(([x,y],[u,v]) => [x+u/points.length,y+v/points.length],[0,0]);
}
function alignedInput(points) {
  const [sx,sy] = centroid(points), [tx,ty] = centroid(reference);
  let dot = 0, cross = 0, energy = 0;
  points.forEach(([x,y], i) => {
    const dx = x-sx, dy = y-sy, u = reference[i][0]-tx, v = reference[i][1]-ty;
    dot += dx*u + dy*v;
    cross += dx*v - dy*u;
    energy += dx*dx + dy*dy;
  });
  if (!(energy > 0)) throw new Error('Invalid face landmarks');
  const a = dot/energy, b = cross/energy;
  crop.fillStyle = 'black';
  crop.fillRect(0,0,112,112);
  crop.save();
  crop.setTransform(a,b,-b,a,tx-a*sx+b*sy,ty-b*sx-a*sy);
  crop.drawImage(view,0,0);
  crop.restore();
  const pixels = crop.getImageData(0,0,112,112).data;
  const plane = 112*112, input = new Float32Array(3*plane);
  for (let c=0; c<3; c++) {
    for (let i=0; i<plane; i++) input[c*plane+i] = pixels[4*i+c]/127.5-1;
  }
  return new ort.Tensor('float32', input, [1,3,112,112]);
}
async function analyze(source, id) {
  const sw = source.videoWidth || source.naturalWidth, sh = source.videoHeight || source.naturalHeight;
  const ratio = Math.min(1,960/Math.max(sw,sh));
  view.width = Math.max(1,Math.round(sw*ratio)); view.height = Math.max(1,Math.round(sh*ratio));
  const w = view.width, h = view.height, scale = 640/Math.max(w,h);
  ctx.drawImage(source,0,0,w,h);
  if (source === video) {
    view.classList.toggle('selfie', video.classList.contains('selfie'));
    view.hidden = false;
    video.hidden = true;
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    capture.hidden = true;
    retake.hidden = false;
  }
  const [detector, classifier] = await models();
  if (id !== generation) return;
  detectorInput.fillStyle = 'black'; detectorInput.fillRect(0,0,640,640);
  detectorInput.drawImage(view,0,0,w*scale,h*scale);
  const pixels = detectorInput.getImageData(0,0,640,640).data;
  const plane = 640*640, data = new Float32Array(3*plane);
  for (let i=0;i<plane;i++) {
    data[i]=pixels[4*i+2]; data[i+plane]=pixels[4*i+1]; data[i+2*plane]=pixels[4*i];
  }
  const outputs = await detector.run({input:new ort.Tensor('float32',data,[1,3,640,640])});
  if (id !== generation) return;
  const face = selectFace(outputs,scale,w,h);
  view.hidden = false;
  if (!face) {
    status.textContent = 'No face detected. Try a closer, well-lit portrait.';
    result.hidden = true;
    return;
  }
  const prediction = await classifier.run({data: alignedInput(face.points)});
  if (id !== generation) return;
  const logits = prediction.label.data;
  const peak = Math.max(...logits);
  // Temperature fitted on 249 detected portraits sampled from FER-2013 PublicTest.
  const temperature = 3.641778;
  const exp = logits.map(value => Math.exp((value-peak)/temperature));
  const sum = exp.reduce((a,b) => a+b,0);
  const rows = labels.map((label, i) => ({label, value: exp[i] / sum}))
    .sort((a, b) => b.value - a.value);
  ctx.strokeStyle='#ffad84'; ctx.lineWidth=Math.max(2,w/200); ctx.strokeRect(...face.box);
  $('ol').replaceChildren(...rows.map(({label,value})=>{
    const row=document.createElement('li'), meter=document.createElement('meter');
    meter.value=value; meter.setAttribute('aria-label',label);
    row.append(label,meter,Math.round(100*value)+'%'); return row;
  }));
  status.textContent = rows[0].value < .45
    ? 'Uncertain expression. Several labels fit this face.'
    : 'Showing scores for the outlined face.';
  result.hidden=false;
}
function fail(id,error) {
  if(id!==generation) return;
  reset();
  console.error(error);
  status.textContent=error.name==='NotAllowedError'?'Camera access was denied. Allow the camera in browser settings or choose an image.':
    'Could not analyze this input. Check your connection, try a JPEG or PNG portrait, or open the link in an up-to-date browser.';
}
async function openCamera() {
  const id = reset();
  camera.disabled = true;
  status.textContent = 'Opening camera…';
  try {
    const media = await navigator.mediaDevices.getUserMedia({
      video: {facingMode: 'user', width: {ideal: 640}, frameRate: {ideal: 24, max: 30}},
      audio: false,
    });
    if (id !== generation) {
      media.getTracks().forEach(track => track.stop());
      return;
    }
    stream = media;
    const facing = media.getVideoTracks()[0].getSettings().facingMode;
    video.classList.toggle('selfie', facing === 'user' || !facing);
    video.srcObject = media;
    await video.play();
    if (id !== generation) return;
    video.hidden = false;
    camera.hidden = true;
    capture.hidden = false;
    status.textContent = 'Center your face in the preview, then take a picture.';
    models().catch(() => {}); // Capture retries if the initial download fails.
  } catch (error) { fail(id, error); }
}
camera.onclick = retake.onclick = openCamera;
capture.onclick = async () => {
  if (!stream) return;
  const id = generation;
  status.textContent = 'Analyzing picture…';
  try { await analyze(video, id); }
  catch (error) { fail(id, error); }
};
$('#upload').onchange=async event=>{
  const file=event.target.files[0];
  if(!file)return;
  event.target.value='';
  const id=reset(), url=URL.createObjectURL(file);
  status.textContent='Loading model and analyzing portraitâ€¦';
  try {
    const photo=new Image(); photo.src=url; await photo.decode();
    await analyze(photo,id);
  } catch(error){fail(id,error);} finally{URL.revokeObjectURL(url);}
};
window.addEventListener('pagehide',reset);
document.addEventListener('visibilitychange',()=>{
  if(document.hidden && (stream || camera.disabled && !camera.hidden)){reset();status.textContent='Camera stopped.';}
});

