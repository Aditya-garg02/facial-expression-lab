const $ = selector => document.querySelector(selector);
const video = $('video'), view = $('canvas'), ctx = view.getContext('2d');
const status = $('#status'), camera = $('#camera'), result = $('#result');
const labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
function surface(size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  return canvas.getContext('2d', {willReadFrequently: true});
}
const detectorInput = surface(640), crop = surface(48);
let runtime, stream, generation = 0, displayedScores;
const scoreMemory = new Float64Array(labels.length);
let lastScoreUpdate = 0;
function models() {
  if (!runtime) {
    ort.env.wasm.numThreads = 1;
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
  displayedScores = undefined;
  lastScoreUpdate = 0;
  camera.textContent = 'Start live camera';
  result.hidden = view.hidden = true;
  return generation;
}
function largestFace(outputs, scale, w, h) {
  let best, area = 0;
  for (const stride of [8,16,32]) {
    const cls = outputs['cls_' + stride].data, obj = outputs['obj_' + stride].data;
    const boxes = outputs['bbox_' + stride].data, columns = 640 / stride;
    for (let i = 0; i < cls.length; i++) {
      if (Math.sqrt(cls[i] * obj[i]) < .8) continue;
      const k = i * 4;
      const cx = (i % columns + boxes[k]) * stride / scale;
      const cy = (Math.floor(i / columns) + boxes[k+1]) * stride / scale;
      const bw = Math.exp(boxes[k+2]) * stride / scale, bh = Math.exp(boxes[k+3]) * stride / scale;
      const x = Math.max(0, cx-bw/2), y = Math.max(0, cy-bh/2);
      const width = Math.min(w,cx+bw/2)-x, height = Math.min(h,cy+bh/2)-y;
      if (width > 0 && height > 0 && width*height > area) {
        area = width*height;
        best = [x,y,width,height];
      }
    }
  }
  return best;
}
async function analyze(source, live, id) {
  const [detector, classifier, profile] = await models();
  if (id !== generation) return;
  const sw = source.videoWidth || source.naturalWidth, sh = source.videoHeight || source.naturalHeight;
  const ratio = Math.min(1,960/Math.max(sw,sh));
  view.width = Math.max(1,Math.round(sw*ratio)); view.height = Math.max(1,Math.round(sh*ratio));
  const w = view.width, h = view.height, scale = 640/Math.max(w,h);
  ctx.drawImage(source,0,0,w,h);
  detectorInput.fillStyle = 'black'; detectorInput.fillRect(0,0,640,640);
  detectorInput.drawImage(view,0,0,w*scale,h*scale);
  const pixels = detectorInput.getImageData(0,0,640,640).data;
  const plane = 640*640, data = new Float32Array(3*plane);
  for (let i=0;i<plane;i++) {
    data[i]=pixels[4*i+2]; data[i+plane]=pixels[4*i+1]; data[i+2*plane]=pixels[4*i];
  }
  const outputs = await detector.run({input:new ort.Tensor('float32',data,[1,3,640,640])});
  if (id !== generation) return;
  const box = largestFace(outputs,scale,w,h);
  view.hidden = false;
  if (!box) {
    status.textContent = 'No face detected. Try a closer, well-lit portrait.';
    result.hidden = true;
    return;
  }
  const [x,y,bw,bh] = box;
  const left=Math.max(0,x-.1*bw), top=Math.max(0,y-.1*bh);
  crop.drawImage(view,left,top,Math.min(w,x+1.1*bw)-left,Math.min(h,y+1.1*bh)-top,0,0,48,48);
  const rgb=crop.getImageData(0,0,48,48).data, input=new Float32Array(48*48);
  for(let i=0;i<input.length;i++) input[i]=(.299*rgb[i*4]+.587*rgb[i*4+1]+.114*rgb[i*4+2])/127.5-1;
  const score = async values => (await classifier.run({face:new ort.Tensor('float32',values,[1,1,48,48])})).scores.data;
  const logits = await score(input);
  if (!live && profile.selected==='mirrored') {
    const mirror = input.map((_,i)=>input[i-i%48+47-i%48]);
    const other = await score(mirror);
    for(let i=0;i<logits.length;i++) logits[i]=(logits[i]+other[i])/2;
  }
  if (id !== generation) return;
  const peak=Math.max(...logits), temp=live?profile.fast_temperature:profile.temperature;
  const exp=logits.map(value=>Math.exp((value-peak)/temp)), sum=exp.reduce((a,b)=>a+b,0);
  const now = performance.now();
  if (!live || !displayedScores) {
    displayedScores = labels.map((_,i)=>exp[i]/sum);
    lastScoreUpdate = now;
  } else {
    const elapsed = Math.max(0, now-lastScoreUpdate);
    const blend = 1-Math.exp(-elapsed/1800);
    for (let i=0;i<labels.length;i++) scoreMemory[i]=displayedScores[i]+(exp[i]/sum-displayedScores[i])*blend;
    const total = scoreMemory.reduce((a,b)=>a+b,0);
    displayedScores = Array.from(scoreMemory,value=>value/total);
    lastScoreUpdate = now;
  }
  const rows=labels.map((label,i)=>({label,value:displayedScores[i]})).sort((a,b)=>b.value-a.value);
  ctx.strokeStyle='#ffad84'; ctx.lineWidth=Math.max(2,w/200); ctx.strokeRect(...box);
  $('ol').replaceChildren(...rows.map(({label,value})=>{
    const row=document.createElement('li'), meter=document.createElement('meter');
    meter.value=value; meter.setAttribute('aria-label',label);
    row.append(label,meter,Math.round(100*value)+'%'); return row;
  }));
  status.textContent='Showing scores for the outlined face.';
  result.hidden=false;
}
function fail(id,error) {
  if(id!==generation) return;
  reset();
  console.error(error);
  status.textContent=error.name==='NotAllowedError'?'Camera access was denied. Allow the camera in browser settings or choose an image.':
    'Could not analyze this input. Check your connection, try a JPEG or PNG portrait, or open the link in an up-to-date browser.';
}
camera.onclick=async()=>{
  const stopping=camera.textContent==='Stop camera';
  const id=reset();
  if(stopping){status.textContent='Camera stopped.';return;}
  camera.textContent='Stop camera';
  status.textContent='Opening camera and loading the model…';
  try {
    const media=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640}},audio:false});
    if(id!==generation){media.getTracks().forEach(track=>track.stop());return;}
    stream=media; video.srcObject=media; await video.play();
    while(id===generation){
      await analyze(video,true,id);
      await new Promise(resolve=>setTimeout(resolve,700));
    }
  } catch(error){fail(id,error);}
};
$('#upload').onchange=async event=>{
  const file=event.target.files[0];
  if(!file)return;
  event.target.value='';
  const id=reset(), url=URL.createObjectURL(file);
  status.textContent='Loading model and analyzing portrait…';
  try {
    const photo=new Image(); photo.src=url; await photo.decode();
    await analyze(photo,false,id);
  } catch(error){fail(id,error);} finally{URL.revokeObjectURL(url);}
};
window.addEventListener('pagehide',reset);
document.addEventListener('visibilitychange',()=>{
  if(document.hidden && camera.textContent==='Stop camera'){reset();status.textContent='Camera stopped.';}
});
