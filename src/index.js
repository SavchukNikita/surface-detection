import jsfeat from './libs/jsfeat';

const BLUR_RADIUS = 4;
const U_MAX = new Int32Array([15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3,0]);
const IMG_WIDTH = 640;
const IMG_HEIGHT = 480;
const TRAIN_LVLS = 4;


const video = document.createElement('video');
const canvas = document.getElementById('canvas');
const canvasCtx = canvas.getContext('2d');

let screenCorners = [];
let screenDescriptors = new jsfeat.matrix_t(32, 500, jsfeat.U8_t | jsfeat.C1_t);
let matches = [];
let homography3Matrix = new jsfeat.matrix_t(3,3,jsfeat.F32C1_t);
let matchMask = new jsfeat.matrix_t(500,1,jsfeat.U8C1_t);
let patternDescriptors = [];
let patternCorners = [];

let imgU8;
let patternPeview;

const initApp = () => {
  canvas.width = IMG_WIDTH;
  canvas.height = IMG_HEIGHT;

  jsfeat.fast_corners.set_threshold(15)

  for (let i = 0; i < IMG_WIDTH*IMG_HEIGHT; i += 1) {
    screenCorners.push(new jsfeat.keypoint_t(0,0,0,0,-1));
    matches[i] = {
      screen_idx: 0,
      pattern_lev: 0,
      pattern_idx: 0,
      distance: 0,
    }
  }

  tick();
}

const tick = () => {
  window.requestAnimationFrame(tick);
  canvasCtx.drawImage(video, 0, 0);

  if (video.readyState !== video.HAVE_ENOUGH_DATA) return;
  
  imgU8 = new jsfeat.matrix_t(IMG_WIDTH, IMG_HEIGHT, jsfeat.U8_t | jsfeat.C1_t);
  let imageData = canvasCtx.getImageData(0, 0, IMG_WIDTH, IMG_HEIGHT);

  jsfeat.imgproc.grayscale(imageData.data, IMG_WIDTH, IMG_HEIGHT, imgU8);
  let imgU8Blured = Object.assign(imgU8); 
  jsfeat.imgproc.gaussian_blur(imgU8, imgU8Blured, BLUR_RADIUS);

  let cornersNum = jsfeat.fast_corners.detect(imgU8Blured, screenCorners, 3);

  jsfeat.orb.describe(imgU8Blured, screenCorners, cornersNum, screenDescriptors);
  let dataU32 = new Uint32Array(imageData.data.buffer);
  renderCorners(screenCorners, cornersNum, dataU32, IMG_WIDTH);

  let numMatches = 0;
  let goodMatches = 0;

  if (patternPeview) {
    renderPatternImg(patternPeview.data, dataU32, patternPeview.cols, patternPeview.rows, IMG_WIDTH);
    numMatches = matchPattern();
    goodMatches = findTransform(matches, numMatches);
  }

  canvasCtx.putImageData(imageData, 0, 0);

  if (numMatches) {
    renderMatches(canvasCtx, matches, numMatches);
    renderSurfaceShape();
  }
}

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

      for(let lev = 0; lev < TRAIN_LVLS; lev += 1) {
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

function popcnt32(n) {
  n -= ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4))& 0xF0F0F0F)* 0x1010101) >> 24;
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

function findTransform(matches, count) {
  let motionModelKernel = new jsfeat.motion_model.homography2d();
  let ransacParam = new jsfeat.ransac_params_t(4, 3, 0.5, 0.99);

  let patternCoords = [];
  let screenCoords = [];

  for (let i = 0; i < count; i += 1) {
      let m = matches[i];
      let screenPoint = screenCorners[m.screen_idx];
      let patternPoint = patternCorners[m.pattern_lev][m.pattern_idx];
      patternCoords[i] = {"x":patternPoint.x, "y":patternPoint.y};
      screenCoords[i] =  {"x":screenPoint.x, "y":screenPoint.y};
  }

  let isSuccessRansac = jsfeat.motion_estimator.ransac(ransacParam,motionModelKernel, patternCoords, screenCoords, count, homography3Matrix, matchMask, 1000);

  let goodPointsCount = 0;
  if (isSuccessRansac) {
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

let startbox = {x:[],y:[]};
let linesbox = [{x:[],y:[]}, {x:[],y:[]},{x:[],y:[]}, {x:[],y:[]}]

const renderSurfaceShape = () => {
  let coords = [];
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
  }

  canvasCtx.lineTo(average(linesbox[0].x), average(linesbox[0].y));
  canvasCtx.lineWidth = 4;
  canvasCtx.stroke();
}

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

function render_pattern_shape(ctx) {
  // get the projected pattern corners
  var coords = [];
  var shape_pts = tCorners(homo3x3.data, pattern_preview.cols*2, pattern_preview.rows*2);
  ctx.strokeStyle = "rgb(0,0,255)";
  ctx.beginPath();
  //E: just a simple adjustment of the box, by using moving averages
  var intersect = false;
  if (lineSegmentsIntersect(shape_pts[0].x,shape_pts[0].y, shape_pts[1].x,shape_pts[1].y, shape_pts[2].x,shape_pts[2].y, shape_pts[3].x,shape_pts[3].y)) {
    intersect = true;
  };
  if (lineSegmentsIntersect(shape_pts[1].x,shape_pts[1].y, shape_pts[2].x,shape_pts[2].y, shape_pts[3].x,shape_pts[3].y, shape_pts[0].x,shape_pts[0].y)) {
    intersect = true;
  };
  
  const avesize = 20;
  
  function cases(v,l){
      let vv;
      switch(v){
          case v < average(l) - 25:
              vv = average(l) - 25;
              break;
          case v > average(l) + 25:
              vv =  average(l) + 25;
              break;
          default:
              vv = v;
      };
      return vv;
  }
  
  if (!intersect) {
    for (let i = 0; i < 4; ++i){
        if (i === 0) {
          if (startbox.x.length < avesize) {
            startbox.x.push(shape_pts[i].x)
          }else{
            let xvertexs = shape_pts[i].x;
            startbox.x.shift()
            //console.log(xvertexs)
            if(xvertexs < average(startbox.x) - 25){
              xvertexs = average(startbox.x) - 25;
            }else if (xvertexs > average(startbox.x) + 25) {
              xvertexs = average(startbox.x) + 25;
            };
            //console.log(xvertexs)
            startbox.x.push(xvertexs)
          };
          if (startbox.y.length < avesize) {
            startbox.y.push(shape_pts[i].y)
          }else{
            let yvertexs = shape_pts[i].y;
            startbox.y.shift()
            //console.log(yvertexs)
            if(yvertexs < average(startbox.y) - 25){
              yvertexs = average(startbox.y) - 25;
            }else if (yvertexs > average(startbox.y) + 25) {
              yvertexs = average(startbox.y) + 25;
            };
            //console.log(yvertexs)
            startbox.y.push(yvertexs)
          }
        };
        if (linesbox[i].x.length < avesize) {
          linesbox[i].x.push(shape_pts[i].x)
        }else{
          let xvertex = cases(shape_pts[i].x, linesbox[i].x);
          linesbox[i].x.shift()
            if(xvertex < average(linesbox[i].x) - 25){
              xvertex = average(linesbox[i].x) - 25;
            }else if (xvertex > average(linesbox[i].x) + 25) {
              xvertex = average(linesbox[i].x) + 25;
            };
          linesbox[i].x.push(xvertex)
        };
        if (linesbox[i].y.length < avesize) {
          linesbox[i].y.push(shape_pts[i].y)
        }else{
          let yvertex = cases(shape_pts[i].y, linesbox[i].y);
          linesbox[i].y.shift()
            if(yvertex < average(linesbox[i].y) - 25){
              yvertex = average(linesbox[i].y) - 25;
            }else if (yvertex > average(linesbox[i].y) + 25) {
              yvertex = average(linesbox[i].y) + 25;
            };
          linesbox[i].y.push(yvertex)
        }
    };
  }

  let findminmaxX = [];
  let findminmaxY = [];
  ctx.moveTo(average(startbox.x),average(startbox.y));
  for(let i = 1; i < 4; ++i){
    ctx.lineTo(average(linesbox[i].x),average(linesbox[i].y));
    findminmaxX.push(average(linesbox[i].x));
    findminmaxY.push(average(linesbox[i].y));
  }
  ctx.lineTo(average(linesbox[0].x),average(linesbox[0].y));
  findminmaxX.push(average(linesbox[0].x));
  findminmaxY.push(average(linesbox[0].y));
  ctx.lineWidth=4;
  ctx.stroke();
  var coords = [];
  //console.log(Math.min(...findminmaxX));
  coords.push(Math.min(...findminmaxX));
  coords.push(Math.min(...findminmaxY));
  coords.push(Math.max(...findminmaxX));
  coords.push(Math.max(...findminmaxY));
  COORDS = []
  COORDS.push(coords)
  //aFrame.coords = coords;
  
}

const setTrainImage = () => {
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

  for(let lev=0; lev < TRAIN_LVLS; lev += 1) {
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

  jsfeat.imgproc.gaussian_blur(levBaseImg, levImg, BLUR_RADIUS);
  cornersNum = jsfeat.fast_corners.detect(levImg, levCorners, 5);
  jsfeat.orb.describe(levImg, levCorners, cornersNum, levDescriptors);

  sc /= scBase;

  for(let lev = 1; lev < TRAIN_LVLS; lev += 1) {
      levCorners = patternCorners[lev];
      levDescriptors = patternDescriptors[lev];

      nWidth = (levBaseImg.cols*sc)|0;
      nHeight = (levBaseImg.rows*sc)|0;

      jsfeat.imgproc.resample(levBaseImg, levImg, nWidth, nHeight);
      jsfeat.imgproc.gaussian_blur(levImg, levImg, BLUR_RADIUS);
      cornersNum = jsfeat.fast_corners.detect(levImg, levCorners, 5);
      jsfeat.orb.describe(levImg, levCorners, cornersNum, levDescriptors);

      for(i = 0; i < cornersNum; ++i) {
          levCorners[i].x *= 1./sc;
          levCorners[i].y *= 1./sc;
      }

      sc /= scBase;
  }
};

// утилиты

const average = (arr) => {
  if (!arr.length) return;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  });

video.addEventListener('loadeddata', initApp);

const trainBtm = document.getElementById('trainBtn');
trainBtm.addEventListener('click', setTrainImage)

navigator.mediaDevices.enumerateDevices().then((list) => console.log(list));



// const detectKeypoints = (img, corners) => {
//   let count = jsfeat.yape06.detect(img, corners, 17);

//   for(let i = 0; i < count; ++i) {
//       corners[i].angle = icAngle(img, corners[i].x, corners[i].y);
//   }

//   return count;
// }

// const icAngle = (img, px, py) => {
//   let m_01 = 0, m_10 = 0;
//   let src = img.data, step = img.cols;
//   let u = 0, v = 0, center_off = (py*step + px) | 0;
//   let v_sum = 0, d = 0, val_plus = 0, val_minus = 0;

//   for (u = -U_MAX.length; u <= U_MAX.length; ++u)
//       m_10 += u * src[center_off+u];

//   for (v = 1; v <= U_MAX.length; ++v) {
//       v_sum = 0;
//       d = U_MAX[v];
//       for (u = -d; u <= d; ++u) {
//           val_plus = src[center_off+u+v*step];
//           val_minus = src[center_off+u-v*step];
//           v_sum += (val_plus - val_minus);
//           m_10 += u * (val_plus + val_minus);
//       }
//       m_01 += v * v_sum;
//   }

//   return Math.atan2(m_01, m_10);
// }