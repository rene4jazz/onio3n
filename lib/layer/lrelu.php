<?php

function forwardLReLU($X, &$layer, $training_mode = false) {
  $Y = [];
  foreach($X as $x){
    $Y[] = ($x > 0.0) ? $x : 0.01;
  }
  if ($training_mode) {
    $layer['grad'] = ['output' => $Y, 'delta' => []];
  }
  return $Y;
}

function backwardLReLU($delta, &$layer, $lr) {
  $Y = [];
  $outputs = $layer['grad']['output'];
  foreach ($delta as $i => $x) { // computes derivate and applies chain rule
    $Y[$i] = $outputs[$i] * $x;
  }
  $layer['grad']['delta'] = $Y;
  return $Y;
}
