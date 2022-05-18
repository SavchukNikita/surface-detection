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
let pattern_corners = [];

let imgU8;
let pattern_preview;

const initApp = () => {
  canvas.width = IMG_WIDTH;
  canvas.height = IMG_HEIGHT;

  for (let i = 0; i < IMG_WIDTH*IMG_HEIGHT; i += 1) {
    screenCorners.push(new jsfeat.keypoint_t(0,0,0,0,-1));
    matches[i] = new match_t();
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

  jsfeat.yape06.laplacian_threshold = 30;
  jsfeat.yape06.min_eigen_value_threshold = 25;

  let cornersNum = jsfeat.yape06.detect(imgU8Blured, screenCorners, 17);

  jsfeat.orb.describe(imgU8Blured, screenCorners, cornersNum, screenDescriptors);

  let dataU32 = new Uint32Array(imageData.data.buffer);
  render_corners(screenCorners, cornersNum, dataU32, IMG_WIDTH);

  let num_matches = 0;
  let good_matches = 0;

  if (pattern_preview) {
    renderPatternImg(pattern_preview.data, dataU32, pattern_preview.cols, pattern_preview.rows, 640);
    num_matches = match_pattern();
    good_matches = find_transform(matches, num_matches);

    console.log(num_matches)
  }

  canvasCtx.putImageData(imageData, 0, 0)
  render_matches(canvasCtx, matches, num_matches);
}

const match_pattern = () => {
  var q_cnt = screenDescriptors.rows;
  var query_du8 = screenDescriptors.data;
  var query_u32 = screenDescriptors.buffer.i32; // cast to integer buffer
  var qd_off = 0;
  var qidx=0,lev=0,pidx=0,k=0;
  var num_matches = 0;

  for(qidx = 0; qidx < q_cnt; ++qidx) {
      var best_dist = 256;
      var best_dist2 = 256;
      var best_idx = -1;
      var best_lev = -1;

      for(lev = 0; lev < TRAIN_LVLS; ++lev) {
          var lev_descr = patternDescriptors[lev];
          var ld_cnt = lev_descr.rows;
          var ld_i32 = lev_descr.buffer.i32; // cast to integer buffer
          var ld_off = 0;

          for(pidx = 0; pidx < ld_cnt; ++pidx) {

              var curr_d = 0;
              // our descriptor is 32 bytes so we have 8 Integers
              for(k=0; k < 8; ++k) {
                  curr_d += popcnt32( query_u32[qd_off+k]^ld_i32[ld_off+k] );
              }

              if(curr_d < best_dist) {
                  best_dist2 = best_dist;
                  best_dist = curr_d;
                  best_lev = lev;
                  best_idx = pidx;
              } else if(curr_d < best_dist2) {
                  best_dist2 = curr_d;
              }

              ld_off += 8; // next descriptor
          }
      }

      // filter out by some threshold
      // if(best_dist < 5) {
      //     matches[num_matches].screen_idx = qidx;
      //     matches[num_matches].pattern_lev = best_lev;
      //     matches[num_matches].pattern_idx = best_idx;
      //     num_matches++;
      // }
      

      if(best_dist < 0.8*best_dist2) {
          matches[num_matches].screen_idx = qidx;
          matches[num_matches].pattern_lev = best_lev;
          matches[num_matches].pattern_idx = best_idx;
          num_matches++;
      }

      qd_off += 8; // next query descriptor
  }

  return num_matches;
}

function popcnt32(n) {
  n -= ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4))& 0xF0F0F0F)* 0x1010101) >> 24;
}

const renderPatternImg = (src, dst, sw, sh, dw) => {
  var alpha = (0xff << 24);
  for(var i = 0; i < sh; ++i) {
      for(var j = 0; j < sw; ++j) {
          var pix = src[i*sw+j];
          dst[i*dw+j] = alpha | (pix << 16) | (pix << 8) | pix;
      }
  }
}

const render_matches = (ctx, matches, count) => {
  for(var i = 0; i < count; ++i) {
      var m = matches[i];
      var s_kp = screenCorners[m.screen_idx];
      var p_kp = pattern_corners[m.pattern_lev][m.pattern_idx];
      ctx.strokeStyle = "rgb(0,255,0)";
      ctx.beginPath();
      ctx.moveTo(s_kp.x,s_kp.y);
      ctx.lineTo(p_kp.x*0.5, p_kp.y*0.5);
      ctx.lineWidth=1;
      ctx.stroke();
  }
}

const render_corners = (corners, count, img, step) => {
  var pix = (0xff << 24) | (0x00 << 16) | (0xff << 8) | 0x00;
  
  for(var i=0; i < count; i+=1)
  {
      var x = corners[i].x;
      var y = corners[i].y;
      var off = (x + y * step);
      img[off] = pix;
  }
}

var match_t = (function () {
  function match_t(screen_idx, pattern_lev, pattern_idx, distance) {
      if (typeof screen_idx === "undefined") { screen_idx=0; }
      if (typeof pattern_lev === "undefined") { pattern_lev=0; }
      if (typeof pattern_idx === "undefined") { pattern_idx=0; }
      if (typeof distance === "undefined") { distance=0; }

      this.screen_idx = screen_idx;
      this.pattern_lev = pattern_lev;
      this.pattern_idx = pattern_idx;
      this.distance = distance;
  }
  return match_t;
})();

function find_transform(matches, count) {
  // motion kernel
  var mm_kernel = new jsfeat.motion_model.homography2d();
  // ransac params
  var num_model_points = 4;
  var reproj_threshold = 3;
  var ransac_param = new jsfeat.ransac_params_t(num_model_points,
                                                reproj_threshold, 0.5, 0.99);

  var pattern_xy = [];
  var screen_xy = [];

  // construct correspondences
  for(var i = 0; i < count; ++i) {
      var m = matches[i];
      var s_kp = screenCorners[m.screen_idx];
      var p_kp = pattern_corners[m.pattern_lev][m.pattern_idx];
      pattern_xy[i] = {"x":p_kp.x, "y":p_kp.y};
      screen_xy[i] =  {"x":s_kp.x, "y":s_kp.y};
  }

  // estimate motion
  var ok = false;
  ok = jsfeat.motion_estimator.ransac(ransac_param, mm_kernel,
                                      pattern_xy, screen_xy, count, homography3Matrix, matchMask, 1000);

  // extract good matches and re-estimate
  var good_cnt = 0;
  if(ok) {
      for(var i=0; i < count; ++i) {
          if(matchMask.data[i]) {
              pattern_xy[good_cnt].x = pattern_xy[i].x;
              pattern_xy[good_cnt].y = pattern_xy[i].y;
              screen_xy[good_cnt].x = screen_xy[i].x;
              screen_xy[good_cnt].y = screen_xy[i].y;
              good_cnt++;
          }
      }
      // run kernel directly with inliers only
      mm_kernel.run(pattern_xy, screen_xy, homography3Matrix, good_cnt);
  } else {
      jsfeat.matmath.identity_3x3(homography3Matrix, 1.0);
  }

  return good_cnt;
}

const setTrainImage = () => {
  console.log('hii')
  var lev=0, i=0;
  var sc = 1.0;
  var max_pattern_size = 512;
  var max_per_level = 300;
  var sc_inc = Math.sqrt(2.0); // magic number ;)
  var lev0_img = new jsfeat.matrix_t(imgU8.cols, imgU8.rows, jsfeat.U8_t | jsfeat.C1_t);
  var lev_img = new jsfeat.matrix_t(imgU8.cols, imgU8.rows, jsfeat.U8_t | jsfeat.C1_t);
  var new_width=0, new_height=0;
  var lev_corners, lev_descr;
  var corners_num=0;

  var sc0 = Math.min(max_pattern_size/imgU8.cols, max_pattern_size/imgU8.rows);
  new_width = (imgU8.cols*sc0)|0;
  new_height = (imgU8.rows*sc0)|0;

  jsfeat.imgproc.resample(imgU8, lev0_img, new_width, new_height);

  // prepare preview
  pattern_preview = new jsfeat.matrix_t(new_width>>1, new_height>>1, jsfeat.U8_t | jsfeat.C1_t);
  jsfeat.imgproc.pyrdown(lev0_img, pattern_preview);

  for(lev=0; lev < TRAIN_LVLS; ++lev) {
      pattern_corners[lev] = [];
      lev_corners = pattern_corners[lev];

      // preallocate corners array
      i = (new_width*new_height) >> lev;
      while(--i >= 0) {
          lev_corners[i] = new jsfeat.keypoint_t(0,0,0,0,-1);
      }

      patternDescriptors[lev] = new jsfeat.matrix_t(32, max_per_level, jsfeat.U8_t | jsfeat.C1_t);
  }

  // do the first level
  lev_corners = pattern_corners[0];
  lev_descr = patternDescriptors[0];

  jsfeat.imgproc.gaussian_blur(lev0_img, lev_img, BLUR_RADIUS);
  corners_num = jsfeat.yape06.detect(lev_img, lev_corners, 5);
  jsfeat.orb.describe(lev_img, lev_corners, corners_num, lev_descr);

  console.log("train " + lev_img.cols + "x" + lev_img.rows + " points: " + corners_num);

  sc /= sc_inc;

  // lets do multiple scale levels
  // we can use Canvas context draw method for faster resize
  // but its nice to demonstrate that you can do everything with jsfeat
  for(lev = 1; lev < TRAIN_LVLS; ++lev) {
      lev_corners = pattern_corners[lev];
      lev_descr = patternDescriptors[lev];

      new_width = (lev0_img.cols*sc)|0;
      new_height = (lev0_img.rows*sc)|0;

      jsfeat.imgproc.resample(lev0_img, lev_img, new_width, new_height);
      jsfeat.imgproc.gaussian_blur(lev_img, lev_img, BLUR_RADIUS);
      corners_num = jsfeat.yape06.detect(lev_img, lev_corners, 5);
      jsfeat.orb.describe(lev_img, lev_corners, corners_num, lev_descr);

      // fix the coordinates due to scale level
      for(i = 0; i < corners_num; ++i) {
          lev_corners[i].x *= 1./sc;
          lev_corners[i].y *= 1./sc;
      }

      console.log("train " + lev_img.cols + "x" + lev_img.rows + " points: " + corners_num);

      sc /= sc_inc;
  }
};

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  });

video.addEventListener('loadeddata', initApp);

const trainBtm = document.getElementById('trainBtn');
trainBtm.addEventListener('click', setTrainImage)



// const detectKeypoints = (img, corners) => {
//   var count = jsfeat.yape06.detect(img, corners, 17);

//   for(var i = 0; i < count; ++i) {
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
