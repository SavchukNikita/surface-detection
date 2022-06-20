import AFrame from 'aframe';

AFrame.registerComponent('test', {
  schema: {
    prop: {
      default: 5,
    }
  },
  tick() {
    console.log('aaa');  }
});