var webpack = require("webpack");
var ExtractTextPlugin = require("extract-text-webpack-plugin");
module.exports = require("./webpack.config.js");    // inherit from the main config file

// disable the hot reload
module.exports.entry = [
  "babel-polyfill",
  __dirname + "/" + module.exports.app_root + "/index.jsx"
];

// production env
module.exports.plugins.push(
  new webpack.DefinePlugin({
    "process.env": {
      NODE_ENV: JSON.stringify("production"),
    }
  })
);

// compress the js file
module.exports.plugins.push(
  new webpack.optimize.UglifyJsPlugin({
    comments: false,
    compressor: {
      warnings: false
    }
  })
);

// on errors exit with code
module.exports.plugins.push(
  function() {
    this.plugin("done", function(stats) {
      if (stats.compilation.errors && stats.compilation.errors.length && process.argv.indexOf('--watch') == -1) {
        console.log(stats.compilation.errors);
        process.exit(1);
      }
    })
  }
);

// export css to a separate file
module.exports.module.loaders[1] = {
  test: /\.scss$/,
  loader: ExtractTextPlugin.extract("css!postcss-loader?parser=postcss-scss"),
};

module.exports.plugins.push(
  new ExtractTextPlugin("../css/main.css")
);
