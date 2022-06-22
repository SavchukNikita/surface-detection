import jsfeat from './libs/jsfeat';
import * as dat from 'dat.gui';
import Stats from 'stats.js';
import * as THREE from 'three';

const MAX_ALLOWED_KEYPOINTS = 500;

const options = {
  blurRadius: 4,
  threshold: 20,
  motionEstimation: 4,
  fastRadius: 3,
}

let screenCorners = [];
let screenDescriptors = new jsfeat.matrix_t(32, 500, jsfeat.U8_t | jsfeat.C1_t);
let matches = [];
let homography3Matrix = new jsfeat.matrix_t(3,3,jsfeat.F32C1_t);
let matchMask = new jsfeat.matrix_t(500,1,jsfeat.U8C1_t);
let patternDescriptors = [];
let patternCorners = [];

let imgU8;
let patternPeview;

let canvas = null;
let canvasCtx = null;
let video = null;
let imgWidth = 0;
let imgHeight = 0;
let goodMatchesNum = [];

let COORDS = [];

let outputCoords = [];

let frameCount = 0;

const stats = new Stats();
// main

const start = () => {
  imgWidth = canvas.width;
  imgHeight = canvas.height;

  canvasCtx = canvas.getContext('2d');

  // base options
  jsfeat.fast_corners.set_threshold(options.threshold)

  for (let i = 0; i < imgWidth*imgHeight; i += 1) {
    screenCorners.push(new jsfeat.keypoint_t(0,0,0,0,-1));
    matches[i] = {
      screen_idx: 0,
      pattern_lev: 0,
      pattern_idx: 0,
      distance: 0,
    }
  }

  stats.showPanel(0);
  document.body.appendChild( stats.dom );

  tick();
}

const tick = () => {
  canvasCtx.drawImage(video, 0, 0);

  if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

  stats.begin();
  
  imgU8 = new jsfeat.matrix_t(imgWidth, imgHeight, jsfeat.U8_t | jsfeat.C1_t);
  let imageData = canvasCtx.getImageData(0, 0, imgWidth, imgHeight);

  jsfeat.imgproc.grayscale(imageData.data, imgWidth, imgHeight, imgU8);
  let imgU8Blured = Object.assign(imgU8); 
  jsfeat.imgproc.gaussian_blur(imgU8, imgU8Blured, options.blurRadius);

  let cornersNum = detectKeypoints(imgU8Blured, screenCorners);

  jsfeat.orb.describe(imgU8Blured, screenCorners, cornersNum, screenDescriptors);
  let dataU32 = new Uint32Array(imageData.data.buffer);
  if(guiConfig.showKeypoints) renderCorners(screenCorners, cornersNum, dataU32, imgWidth);

  let numMatches = 0;
  let goodMatches = 0;

  if (patternPeview) {
    renderPatternImg(patternPeview.data, dataU32, patternPeview.cols, patternPeview.rows, imgWidth);
    numMatches = matchPattern();
    goodMatches = findTransform(matches, numMatches);
  }

  canvasCtx.putImageData(imageData, 0, 0);

  if (numMatches) {
    if (guiConfig.showMatches) renderMatches(canvasCtx, matches, numMatches);

    if (goodMatchesNum.length > 20) {
      goodMatchesNum.shift();
    };

    if (goodMatches > 5) {
      goodMatchesNum.push(1);
    } else {
      goodMatchesNum.push(0);
      COORDS = [];
      while(outputCoords.length > 0) {
        outputCoords.pop();
      }
    };

    if (goodMatchesNum.reduce((a, b) => { return a + b}, 0) / goodMatchesNum.length > .5) {
      if (guiConfig.showSurfaceShape) renderSurfaceShape();
    }
  }

  stats.end();
  
  if (frameCount < 120) setTrainImage();
  else frameCount = 0;
  
  window.requestAnimationFrame(tick);
}


//#region [rgba(0, 101, 252, 0.2)]
// detecting

const matchPattern = () => {
  let qCount = screenDescriptors.rows;
  let qU32 = screenDescriptors.buffer.i32;
  let qOff = 0;
  let numMatches = 0;

  for(let qidx = 0; qidx < qCount; qidx += 1) {
      let bestDist = 256;
      let bestDistCurr = 256;
      let bestIdx = -1;
      let bestLev = -1;

      for(let lev = 0; lev < 4; lev += 1) {
          let levDescriptors = patternDescriptors[lev];
          let LDCount = levDescriptors.rows;
          let LDI32 = levDescriptors.buffer.i32;
          let LDOff = 0;

          for(let pidx = 0; pidx < LDCount; ++pidx) {

              let curr = 0;

              for(let k=0; k < 8; ++k) {
                  curr += popcnt32( qU32[qOff+k]^LDI32[LDOff+k] );
              }

              if(curr < bestDist) {
                  bestDistCurr = bestDist;
                  bestDist = curr;
                  bestLev = lev;
                  bestIdx = pidx;
              } else if(curr < bestDistCurr) {
                  bestDistCurr = curr;
              }

              LDOff += 8;
          }
      }

      if(bestDist < options.threshold) {
        matches[numMatches].screen_idx = qidx;
        matches[numMatches].pattern_level = bestLev;
        matches[numMatches].pattern_idx = bestIdx;
        numMatches++;
      }
      

      if(bestDist < 0.8*bestDistCurr) {
          matches[numMatches].screen_idx = qidx;
          matches[numMatches].pattern_lev = bestLev;
          matches[numMatches].pattern_idx = bestIdx;
          numMatches++;
      }

      qOff += 8;
  }

  return numMatches;
}

const findTransform = (matches, count) => {
  let motionModelKernel = new jsfeat.motion_model.homography2d();
  let params = new jsfeat.ransac_params_t(options.motionEstimation, 3, 0.5, 0.99);

  let patternCoords = [];
  let screenCoords = [];

  for (let i = 0; i < count; i += 1) {
      let m = matches[i];
      let screenPoint = screenCorners[m.screen_idx];
      let patternPoint = patternCorners[m.pattern_lev][m.pattern_idx];
      patternCoords[i] = {"x":patternPoint.x, "y":patternPoint.y};
      screenCoords[i] =  {"x":screenPoint.x, "y":screenPoint.y};
  }

  let isSuccess = jsfeat.motion_estimator.lmeds(params,motionModelKernel, patternCoords, screenCoords, count, homography3Matrix, matchMask, 1000);
  
  let goodPointsCount = 0;
  if (isSuccess) {
      for(let i = 0; i < count; i += 1) {
          if(matchMask.data[i]) {
              patternCoords[goodPointsCount].x = patternCoords[i].x;
              patternCoords[goodPointsCount].y = patternCoords[i].y;
              screenCoords[goodPointsCount].x = screenCoords[i].x;
              screenCoords[goodPointsCount].y = screenCoords[i].y;
              goodPointsCount += 1;
          }
      }
      motionModelKernel.run(patternCoords, screenCoords, homography3Matrix, goodPointsCount);
  } else {
      jsfeat.matmath.identity_3x3(homography3Matrix, 1.0);
  }

  return goodPointsCount;
}

const affineTransform = (matrix, w, h) => {
  let pt = [
    { x:0, y: 0},
    { x: w, y: 0},
    { x: w, y: h},
    { x: 0, y: h},
  ];

  let px = 0;
  let py = 0;
  let z = 0;

  for ( let i = 0; i < pt.length; i++ ) {
    px = matrix[0]*pt[i].x + matrix[1]*pt[i].y + matrix[2];
    py = matrix[3]*pt[i].x + matrix[4]*pt[i].y + matrix[5];
    z = matrix[6]*pt[i].x + matrix[7]*pt[i].y + matrix[8];

    pt[i].x = px/z;
    pt[i].y = py/z;
  }

  return pt;
}

const detectKeypoints = (img, corners) => {
  let count = jsfeat.fast_corners.detect(img, corners, options.fastRadius);

  if(count > MAX_ALLOWED_KEYPOINTS) {
    jsfeat.math.qsort(corners, 0, count-1, function (a,b) { return (b.score<a.score) });
    count = MAX_ALLOWED_KEYPOINTS;
  }

  return count;
}



const setTrainImage = () => {
  frameCount += 1;
  let i=0;
  let sc = 1.0;
  let maxPatternSize = 512;
  let maxPerLevel = 300;
  // магическое число
  let scBase = Math.sqrt(2.0);
  let levBaseImg = new jsfeat.matrix_t(imgU8.cols, imgU8.rows, jsfeat.U8_t | jsfeat.C1_t);
  let levImg = new jsfeat.matrix_t(imgU8.cols, imgU8.rows, jsfeat.U8_t | jsfeat.C1_t);
  let nWidth=0
  let nHeight=0;
  let cornersNum=0;
  let sc0 = Math.min(maxPatternSize/imgU8.cols, maxPatternSize/imgU8.rows);

  let levCorners, levDescriptors;

  nWidth = (imgU8.cols*sc0)|0;
  nHeight = (imgU8.rows*sc0)|0;

  jsfeat.imgproc.resample(imgU8, levBaseImg, nWidth, nHeight);

  patternPeview = new jsfeat.matrix_t(nWidth>>1, nHeight>>1, jsfeat.U8_t | jsfeat.C1_t);
  jsfeat.imgproc.pyrdown(levBaseImg, patternPeview);

  for(let lev=0; lev < 4; lev += 1) {
      patternCorners[lev] = [];
      levCorners = patternCorners[lev];

      i = (nWidth*nHeight) >> lev;
      while(--i >= 0) {
          levCorners[i] = new jsfeat.keypoint_t(0,0,0,0,-1);
      }

      patternDescriptors[lev] = new jsfeat.matrix_t(32, maxPerLevel, jsfeat.U8_t | jsfeat.C1_t);
  }

  levCorners = patternCorners[0];
  levDescriptors = patternDescriptors[0];

  jsfeat.imgproc.gaussian_blur(levBaseImg, levImg, options.blurRadius);
  console.log(levCorners);
  cornersNum = detectKeypoints(levImg, levCorners);
  jsfeat.orb.describe(levImg, levCorners, cornersNum, levDescriptors);

  sc /= scBase;

  for(let lev = 1; lev < 4; lev += 1) {
      levCorners = patternCorners[lev];
      levDescriptors = patternDescriptors[lev];

      nWidth = (levBaseImg.cols*sc)|0;
      nHeight = (levBaseImg.rows*sc)|0;

      jsfeat.imgproc.resample(levBaseImg, levImg, nWidth, nHeight);
      jsfeat.imgproc.gaussian_blur(levImg, levImg, options.blurRadius);
      cornersNum = detectKeypoints(levImg, levCorners);
      jsfeat.orb.describe(levImg, levCorners, cornersNum, levDescriptors);

      for(i = 0; i < cornersNum; ++i) {
          levCorners[i].x *= 1./sc;
          levCorners[i].y *= 1./sc;
      }

      sc /= scBase;
  }
}

const getCoords = () => {
  const width = Math.round(60 * imgWidth / imgHeight);
  const height = 60;
  
  // Adjust camera frustum near and far clipping plane to match these distances.
  // E: this is the calibration issue - he is APPROXIMATING it
  // --- the distances he mentions, they are the near and far attributes of the fustrum as in the aframe a-camera element
  const MIN_DETECTED_HEIGHT = 0.3; // ~At about 2.5m~ modified
  const MAX_DETECTED_HEIGHT = 0.8; // ~At about 0.5m~ modified

  const res = [];

  for (let i = 0; i < COORDS.length; i++) {
    let coord = COORDS[i];

    if (isNaN(coord[0])) return;

    const x = 2 * (coord[0] / width + coord[2] / width / 2) - 1;
    const y = 1 - 2 * ((coord[1] / height) - (coord[3] / height) / 2);

    const z = 1 - 2 * ((coord[3] / height) - MIN_DETECTED_HEIGHT) / (MAX_DETECTED_HEIGHT - MIN_DETECTED_HEIGHT);

    if (outputCoords.length > 20) {
      outputCoords.shift();
    };

    outputCoords.push({x, y, z});
  }
}

//#endregion

// region[rgba(168, 0, 252, 0.1)]
// graphics

const renderMatches = (ctx, matches, count) => {
  for(let i = 0; i < count; ++i) {
      let m = matches[i];
      let screenPoint = screenCorners[m.screen_idx];
      let patternPoint = patternCorners[m.pattern_lev][m.pattern_idx];

      ctx.strokeStyle = "rgb(0,255,0)";
      ctx.beginPath();
      ctx.moveTo(screenPoint.x,screenPoint.y);
      ctx.lineTo(patternPoint.x*0.5, patternPoint.y*0.5);
      ctx.lineWidth=1;
      ctx.stroke();
  }
}

const renderPatternImg = (src, dst, sw, sh, dw) => {
  let alpha = (0xff << 24);
  for(let i = 0; i < sh; ++i) {
      for(let j = 0; j < sw; ++j) {
          let pix = src[i*sw+j];
          dst[i*dw+j] = alpha | (pix << 16) | (pix << 8) | pix;
      }
  }
}

const renderCorners = (corners, count, img, step) => {
  let pix = (0xff << 24) | (0x00 << 16) | (0xff << 8) | 0x00;
  
  for(let i=0; i < count; i+=1)
  {
      let x = corners[i].x;
      let y = corners[i].y;
      let off = (x + y * step);
      img[off] = pix;
  }
}

let startbox = {x:[],y:[]};
let linesbox = [{x:[],y:[]}, {x:[],y:[]},{x:[],y:[]}, {x:[],y:[]}]

const renderSurfaceShape = () => {
  let shapePlots = affineTransform(homography3Matrix.data, patternPeview.cols * 2, patternPeview.rows * 2);

  canvasCtx.strokeStyle = "#0000FF"
  canvasCtx.beginPath();

  let intersect = false;

  if (
    isIntersect(shapePlots[0], shapePlots[1], shapePlots[2], shapePlots[3]) ||
    isIntersect(shapePlots[1], shapePlots[2], shapePlots[3], shapePlots[0])
  ) {
        intersect = true;
  }

  const avesize = 20;

  if (!intersect) {
    for (let i = 0; i < 4; i++) {
      if (i == 0) {
        if (startbox.x.length < avesize) {
          startbox.x.push(shapePlots[i].x)
        } else {
          startbox.x.shift();
          let xvertex = cases(shapePlots[i].x, startbox.x);

          startbox.x.push(xvertex);
        }

        if (startbox.y.length < avesize) {
          startbox.y.push(shapePlots[i].y);
        } else {
          startbox.y.shift();
          let yvertexs = cases(shapePlots[i].y, startbox.y);
          startbox.y.push(yvertexs);
        }
      }

      if (linesbox[i].x.length < avesize) {
        linesbox[i].x.push(shapePlots[i].x)
      } else {
        linesbox[i].x.shift();
        let xvertex = cases(shapePlots[i].x, linesbox[i].x);

        linesbox[i].x.push(xvertex);
      }

      if (linesbox[i].y.length < avesize ) {
        linesbox[i].y.push(shapePlots[i].y);
      } else {
        linesbox[i].y.shift();
        let yvertex = cases(shapePlots[i].y, linesbox[i].y);
        linesbox[i].y.push(yvertex);
      }
    }
  }

  let minmaxX = [];
  let minmaxY = [];

  canvasCtx.moveTo(average(startbox.x), average(startbox.y));
  
  for ( let i = 1; i < 4; i++ ) {
    canvasCtx.lineTo(average(linesbox[i].x), average(linesbox[i].y));
    minmaxX.push(average(linesbox[i].x));
    minmaxY.push(average(linesbox[i].y));
  }

  canvasCtx.lineTo(average(linesbox[0].x), average(linesbox[0].y));

  minmaxX.push(average(linesbox[0].x));
  minmaxY.push(average(linesbox[0].y));

  canvasCtx.lineWidth = 4;
  canvasCtx.stroke();

  let coordsLocal = [];
  coordsLocal.push(Math.min(...minmaxX));
  coordsLocal.push(Math.min(...minmaxY));
  coordsLocal.push(Math.max(...minmaxX));
  coordsLocal.push(Math.max(...minmaxY));

  // COORDS = [];
  COORDS.push(coordsLocal);

  getCoords();
}

//#endregion


//#region[rgba(43, 255, 0, 0.1)]
// utils

const cases = (v, l) => {
  let vv = 0;
  const av = average(l);

  if (v < av - 25) return av - 25;

  if (v > av + 25) return av + 25;

  return v;
}

const isIntersect = (c1, c2, c3, c4) => {
  var aDx = Math.abs(c2.x - c1.x);
  var aDy = Math.abs(c2.y - c1.y);
  var bDx = Math.abs(c4.x - c3.x);
  var bDy = Math.abs(c4.y - c3.y);
  var s = (-aDy * (c1.x - c3.x) + aDx * (c1.y - c3.y)) / (-bDx * aDy + aDx * bDy);
  var t = (+bDx * (c1.y - c3.y) - bDy * (c1.x - c3.x)) / (-bDx * aDy + aDx * bDy);
  return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

const average = (arr) => {
  if (!arr.length) return;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

const popcnt32 = (n) => {
  n -= ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4))& 0xF0F0F0F)* 0x1010101) >> 24;
}

//#endregion

const guiConfig = {
  blurRadius: { min: 0, max: 10 },
  threshold: { min: 0, max: 128 },
  motionEstimation: { min: 0, max: 8 },
  fastRadius: { min: 0, max: 10 },
  setTrainImage: setTrainImage,
  showKeypoints: true,
  showMatches: true,
  showSurfaceShape: true,
}

const initGUIModule = () => {
  const gui = new dat.GUI();

  for (let op in options) {
    gui.add(options, op, guiConfig[op].min, guiConfig[op].max, 1);
  }

  gui.add(guiConfig, 'showKeypoints');
  gui.add(guiConfig, 'showMatches');
  gui.add(guiConfig, 'showSurfaceShape');

  gui.add(guiConfig, 'setTrainImage');
}

const init = (canvasSelector, gui, coords) => {
  canvas = document.querySelector(canvasSelector);

  if (!canvas) {
    console.error('canvas selector is required');
    return;
  }

  outputCoords = coords;

  video = document.createElement('video');
  video.addEventListener('loadeddata', start);

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    });

  if (gui) initGUIModule(gui);
}

export {
  init
};