export default class Canvas extends HTMLCanvasElement {
  constructor(image, width = 400 ,height = 400) {
    super();
    this.context.drawImage(image, 0, 0);
    this.width = width;
    this.height = height;
  }

  get context() {
    return this.getContext('2d');
  }

  get grayscaleBlurImageData() {
    let imgData = this.context.getImageData();

    let img_u8 = new jsfeat.matrix_t(this.width, this.height, jsfeat.U8_t | jsfeat.C1_t);
    var code = jsfeat.COLOR_RGBA2GRAY;
    
    jsfeat.imgproc.grayscale(imgData.data, this.width, this.height, img_u8, code);

    jsfeat.imgproc.box_blur_gray(img_u8, img_u8, this.BLUR_RADIUS, 0);

    let data_u32 = new Uint32Array(imgData.data.buffer);

    let alpha = (0xff << 24);
    let i = img_u8.cols*img_u8.rows, pix = 0;

    while(--i >= 0) {
        pix = img_u8.data[i];
        data_u32[i] = alpha | (pix << 16) | (pix << 8) | pix;
    }

    return imgData;
  }
}