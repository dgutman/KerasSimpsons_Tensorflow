import React from "react";
import {Router, Route} from "react-router";
import {history} from "./store.js";
import App from './containers/app';

const router = (
  <Router onUpdate={() => window.scrollTo(0, 0)} history={history}>
    <App>
      <Route path="/" component={}/>
    </App>
  </Router>
);

export {router};
