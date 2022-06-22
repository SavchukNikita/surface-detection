import AFrame from 'aframe';
import { init } from '.';

AFrame.registerComponent('test', {
  schema: {
    width: {
      default: 640,
    },
    height: {
      default: 480,
    }
  },
  tick() {
    if (!this.coords.length) {
      for (let i = 0; i < this.entityList.length; i++) {
        this.entityList[i].setAttribute('visible', false);
      }
      return;
    }

    console.log(this.coords);

    for (let i = 0; i < this.coords.length; i++) {
      if (i >= this.entityList.length) {
        return;
      }

      const coord = this.coords[i];
      
      const pos = new THREE.Vector3(coord.x/29, coord.y/29, -coord.z/49).unproject(this.el.camera);
      this.entityList[i].setAttribute('position', pos);
      this.entityList[i].setAttribute('visible', true);
    }
  },
  init() {
    const canvasId = "ar-canvas";
    const canvas = document.createElement('canvas');
    canvas.width = this.data.width;
    canvas.height = this.data.height;
    canvas.id = canvasId;
    this.el.appendChild(canvas);

    let coords = [];

    const entities = document.querySelectorAll('a-entity');
    this.entityList = Array.from(entities).filter((e) => !e.getAttribute('camera'));
    this.coords = coords;

    init(`#${canvasId}`, true, coords);
  }
});