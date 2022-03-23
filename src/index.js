import jsfeat from './libs/jsfeat';

const BLUR_RADIUS = 10;

const video = document.createElement('video');
const canvas = document.createElement('canvas');
const canvasCtx = canvas.getContext('2d');

let canvasWidth = 0;
let canvasHeight = 0;
let screen_corners = [];
let screen_descriptors = {};

document.body.appendChild(canvas);

const initApp = () => {
  canvasWidth = video.videoWidth;
  canvasHeight = video.videoHeight;
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;

  let i = 640*480;
  while(--i >= 0) {
      screen_corners[i] = new jsfeat.keypoint_t(0,0,0,0,-1);
  }

  screen_descriptors = new jsfeat.matrix_t(32, 500, jsfeat.U8_t | jsfeat.C1_t);

  tick();
}

const tick = () => {
  window.requestAnimationFrame(tick);
  canvasCtx.drawImage(video, 0, 0);

  if (video.readyState !== video.HAVE_ENOUGH_DATA) return;
  
  let imgU8 = new jsfeat.matrix_t(canvasWidth, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);
  let imageData = canvasCtx.getImageData(0, 0, canvasWidth, canvasHeight);

  jsfeat.imgproc.grayscale(imageData.data, canvasWidth, canvasHeight, imgU8);
  jsfeat.imgproc.gaussian_blur(imgU8, imgU8, BLUR_RADIUS);

  let num_corners = detectKeypoints(imgU8, screen_corners, 500);

  jsfeat.orb.describe(imgU8, screen_corners, num_corners, screen_descriptors);

  let dataU32 = new Uint32Array(imageData.data.buffer);
  render_corners(screen_corners, num_corners, dataU32, 640);

  canvasCtx.putImageData(imageData, 0, 0)
}

const detectKeypoints = (img, corners) => {
  var count = jsfeat.yape06.detect(img, corners, 17);

  for(var i = 0; i < count; ++i) {
      corners[i].angle = icAngle(img, corners[i].x, corners[i].y);
  }

  return count;
}

let u_max = new Int32Array([15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3,0]);
const icAngle = (img, px, py) => {
  var half_k = 15;
  var m_01 = 0, m_10 = 0;
  var src=img.data, step=img.cols;
  var u=0, v=0, center_off=(py*step + px)|0;
  var v_sum=0,d=0,val_plus=0,val_minus=0;

  // Treat the center line differently, v=0
  for (u = -half_k; u <= half_k; ++u)
      m_10 += u * src[center_off+u];

  // Go line by line in the circular patch
  for (v = 1; v <= half_k; ++v) {
      // Proceed over the two lines
      v_sum = 0;
      d = u_max[v];
      for (u = -d; u <= d; ++u) {
          val_plus = src[center_off+u+v*step];
          val_minus = src[center_off+u-v*step];
          v_sum += (val_plus - val_minus);
          m_10 += u * (val_plus + val_minus);
      }
      m_01 += v * v_sum;
  }

  return Math.atan2(m_01, m_10);
}

const render_corners = (corners, count, img, step) => {
  var pix = (0xff << 24) | (0x00 << 16) | (0xff << 8) | 0x00;
  
  for(var i=0; i < count; ++i)
  {
      var x = corners[i].x;
      var y = corners[i].y;
      var off = (x + y * step);
      img[off] = pix;
      img[off-1] = pix;
      img[off+1] = pix;
      img[off-step] = pix;
      img[off+step] = pix;
  }
}

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  });

video.addEventListener('loadeddata', initApp);