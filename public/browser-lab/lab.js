const $ = selector => document.querySelector(selector);
const video = $('video'), view = $('canvas'), ctx = view.getContext('2d');
const status = $('#status'), camera = $('#camera'), result = $('#result');
const capture = $('#capture'), retake = $('#retake');
const labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
function surface(size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  return canvas.getContext('2d', {willReadFrequently: true});
}
const detectorInput = surface(640), crop = surface(48);
let runtime, stream, generation = 0;
function models() {
  if (!runtime) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = typeof Worker === 'function';
    ort.env.wasm.wasmPaths = new URL('./', location.href).href;
    runtime = Promise.all([
      ort.InferenceSession.create('detector.onnx'),
      ort.InferenceSession.create('expression.onnx'),
      fetch('profile.json').then(response => response.json()),
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
      best = [x, y, boxWidth, boxHeight];
    }
  }
  return best;
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
  const [detector, classifier, profile] = await models();
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
  const box = selectFace(outputs,scale,w,h);
  view.hidden = false;
  if (!box) {
    status.textContent = 'No face detected. Try a closer, well-lit portrait.';
    result.hidden = true;
    return;
  }
  const [x,y,bw,bh] = box;
  const left=Math.max(0,x-.12*bw), top=Math.max(0,y-.12*bh);
  const right=Math.min(w,x+1.12*bw), bottom=Math.min(h,y+1.12*bh);
  crop.drawImage(view,left,top,right-left,bottom-top,0,0,48,48);
  const rgb=crop.getImageData(0,0,48,48).data, input=new Float32Array(48*48);
  for(let i=0;i<input.length;i++) input[i]=(.299*rgb[i*4]+.587*rgb[i*4+1]+.114*rgb[i*4+2])/127.5-1;
  const score = async values => (await classifier.run({face:new ort.Tensor('float32',values,[1,1,48,48])})).scores.data;
  const views = [input];
  if (profile.selected === 'mirrored') {
    views.push(input.map((_, i) => input[i - i % 48 + 47 - i % 48]));
  }
  if (profile.browser_shift_temperature) {
    // Reflect at the crop edge, matching the one-pixel validation transforms.
    for (const original of views.slice()) {
      for (const offset of [-1, 1]) {
        views.push(original.map((_, i) => {
          const x = Math.abs(i % 48 + offset);
          const y = Math.abs(Math.floor(i / 48) + offset);
          return original[Math.min(y, 94 - y) * 48 + Math.min(x, 94 - x)];
        }));
      }
    }
  }
  const logits = new Float32Array(labels.length);
  for (const values of views) {
    if (id !== generation) return;
    const scores = await score(values);
    for (let i = 0; i < labels.length; i++) logits[i] += scores[i] / views.length;
  }
  if (id !== generation) return;
  const peak = Math.max(...logits);
  const temp = profile.browser_shift_temperature ?? profile.temperature;
  const exp=logits.map(value=>Math.exp((value-peak)/temp)), sum=exp.reduce((a,b)=>a+b,0);
  const rows = labels.map((label, i) => ({label, value: exp[i] / sum}))
    .sort((a, b) => b.value - a.value);
  ctx.strokeStyle='#ffad84'; ctx.lineWidth=Math.max(2,w/200); ctx.strokeRect(...box);
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

