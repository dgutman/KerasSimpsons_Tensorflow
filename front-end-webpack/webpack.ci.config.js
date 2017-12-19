var webpack = require("webpack");
var ExtractTextPlugin = require("extract-text-webpack-plugin");
module.exports = require("./webpack.config.js");

// disable the hot reload
module.exports.entry = [
  "babel-polyfill",
  __dirname + "/" + module.exports.app_root + "/index.jsx"
];

// ci env
module.exports.plugins.push(
  new webpack.DefinePlugin({
    "process.env": {
      NODE_ENV: JSON.stringify("ci"),
    }
  })
);
// export css to a separate file
module.exports.module.loaders[1] = {
  test: /\.scss$/,
  loader: ExtractTextPlugin.extract("css!postcss-loader?parser=postcss-scss"),
};

module.exports.plugins.push(
  new ExtractTextPlugin("../css/main.css")
);
