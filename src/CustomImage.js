import jsfeat from './libs/jsfeat';

export default class CustomImage extends Image {
  BLUR_RADIUS = 3;

  constructor(image, w, h) {
    super();
    this.width = w;
    this.height = h;
    this.canvas = document.createElement('canvas');
    this.image = image;

    this.afterLoad();
  }

  get canvasContext() {
    return this.canvas.getContext('2d');
  }

  get imageData() {
    return this.canvasContext.getImageData(0, 0, this.height, this.width);
  }

  canvasDraw() {
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    this.canvasContext.drawImage(this.image, 0, 0, this.width, this.height);
  }

  afterLoad() {
    this.canvasDraw();
    this.canvasContext.putImageData(this.grayscaleBlurImageData, 0, 0);
  }

  get grayscaleBlurImageData() {
    this.canvasDraw();

    let img_u8 = new jsfeat.matrix_t(640, 480, jsfeat.U8_t | jsfeat.C1_t);

    var imageData = this.canvasContext.getImageData(0, 0, 640, 480);

    jsfeat.imgproc.grayscale(imageData.data, 640, 480, img_u8);

    var r = 3;

    jsfeat.imgproc.box_blur_gray(img_u8, img_u8, r, 0);

    // render result back to canvas
    var data_u32 = new Uint32Array(imageData.data.buffer);
    var alpha = (0xff << 24);
    var i = img_u8.cols*img_u8.rows, pix = 0;
    while(--i >= 0) {
        pix = img_u8.data[i];
        data_u32[i] = alpha | (pix << 16) | (pix << 8) | pix;
    }

    return imageData;
  }
}