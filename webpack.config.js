const path = require('path');
const webpack = require('webpack')

module.exports = {
  entry: './src/AFrame.js',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
  },
  devServer: {
    static: {
      directory: path.resolve(__dirname, 'dist'),
    },
    port: 8080,
    open: true,
  },
};